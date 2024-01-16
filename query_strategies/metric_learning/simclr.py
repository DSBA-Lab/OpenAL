'''
code reference
- https://github.com/alinlab/CSI/blob/master/training/sup/sup_simclr_CSI.py
- https://github1s.com/RUC-DWBI-ML/CCAL/blob/main/senmatic_contrast/simclr.py
'''

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from copy import deepcopy
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler

from .factory import MetricLearning
from .losses import SupConLoss
from .transform_layers import get_simclr_aug, TwoCropTransform, HorizontalFlipLayer, Rotation, CutPerm


class SimCLRCSI(MetricLearning):
    def __init__(self, batch_size: int, num_workers: int, shift_trans_type: str = 'rotation', sim_lambda: float = 1.0, **init_params):
        super(SimCLRCSI, self).__init__(**init_params)
        
        self.sim_lambda = sim_lambda
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_criterion = SupConLoss(temperature=0.5)
        
        # get shift transform
        self.get_shift_module(shift_trans_type=shift_trans_type)
    
    def create_datasets(self, dataset, sample_idx: np.ndarray, **kwargs):
        
        # set trainset        
        dataname = dataset.dataname
        trainset = deepcopy(dataset)
        trainset.transform = self.create_transform(dataname=dataname)
        trainloader = DataLoader(trainset, sampler=SubsetRandomSampler(indices=sample_idx), batch_size=self.batch_size, num_workers=self.num_workers)
        
        # simclr_aug
        aug_info = ['ColorJitter', 'RandomGrayscale', 'RandomResizedCrop']
        aug_info = aug_info[:-1] if dataname == 'imagenet' else aug_info
        
        simclr_aug = get_simclr_aug(img_size=kwargs['img_size'], aug_info=aug_info)
        
        # hflip
        hflip = HorizontalFlipLayer()
        
        # num_classes
        num_classes = len(dataset.classes)
        
        # set attributions for trainloader and testset
        setattr(self, 'trainloader', trainloader)
        setattr(self, 'simclr_aug', simclr_aug)
        setattr(self, 'hflip', hflip)
        setattr(self, 'dataname', dataname)
        setattr(self, 'num_classes', num_classes)
        
        
    def create_transform(self, dataname):
        if dataname == 'imagenet':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform = TwoCropTransform(transform)
            
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
        return transform
            
            
    def train(self, vis_encoder, optimizer, device: str):
        total_loss = 0
        
        desc = '[TRAIN] Loss: {loss:>6.4f}'
        p_bar = tqdm(self.trainloader, desc=desc.format(loss=total_loss), leave=False)
        
        self.hflip.to(device)
        
        vis_encoder.train()
        
        for idx, (images, targets) in enumerate(p_bar):          
            # augment images
            if self.dataname != 'imagenet':
                bsz = images.size(0)
                images = images.to(device)
                images1, images2 = self.hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
            else:
                bsz = images[0].size(0)
                images1, images2 = images[0].to(device), images[1].to(device)
            
            images1 = torch.cat([P.shift_trans(images1, k) for k in range(P.K_shift)])
            images2 = torch.cat([P.shift_trans(images2, k) for k in range(P.K_shift)])
            shift_labels = torch.cat([torch.ones_like(targets) * k for k in range(P.K_shift)], 0)  # B -> 4B
            shift_labels = shift_labels.repeat(2).to(device)
            
            images_pair = self.simclr_aug(images_pair)  # simclr augment
            outputs = vis_encoder(images_pair, simclr=True, shift=True)
                       
            f1, f2 = torch.split(outputs['simclr'], [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            # loss
            loss_sim = self.train_criterion(features)
            loss_shift = self.criterion(outputs['shift'], shift_labels)
            loss = loss_sim * self.sim_lambda + loss_shift

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            p_bar.set_description(desc=desc.format(loss=total_loss/(idx+1)))
            
            
    def get_shift_module(self, shift_trans_type: str = 'rotation'):

        if shift_trans_type == 'rotation':
            shift_transform = Rotation()
            k_shift = 4
        elif shift_trans_type == 'cutperm':
            shift_transform = CutPerm()
            k_shift = 4
        else:
            shift_transform = nn.Identity()
            k_shift = 1


        # reduce batch size
        self.batch_size = int(self.batch_size/k_shift)
        
        setattr(self, 'shift_transform', shift_transform)
        setattr(self, 'k_shift', k_shift)
            
            
    
class SimCLR(MetricLearning):
    def __init__(self, batch_size: int, num_workers: int, **init_params):
        super(SimCLRCSI, self).__init__(**init_params)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_criterion = SupConLoss(temperature=0.5)
        
    
    def create_datasets(self, dataset, sample_idx: np.ndarray, **kwargs):
        
        # set trainset        
        dataname = dataset.dataname
        trainset = deepcopy(dataset)
        trainset.transform = self.create_transform(dataname=dataname)
        trainloader = DataLoader(trainset, sampler=SubsetRandomSampler(indices=sample_idx), batch_size=self.batch_size, num_workers=self.num_workers)
        
        # simclr_aug
        aug_info = ['ColorJitter', 'RandomGrayscale', 'RandomResizedCrop']
        aug_info = aug_info[:-1] if dataname == 'imagenet' else aug_info
        
        simclr_aug = get_simclr_aug(img_size=kwargs['img_size'], aug_info=aug_info)
        
        # hflip
        hflip = HorizontalFlipLayer()
        
        # num_classes
        num_classes = len(dataset.classes)
        
        # set attributions for trainloader and testset
        setattr(self, 'trainloader', trainloader)
        setattr(self, 'simclr_aug', simclr_aug)
        setattr(self, 'hflip', hflip)
        setattr(self, 'dataname', dataname)
        setattr(self, 'num_classes', num_classes)
        
        
    def create_transform(self, dataname):
        if dataname == 'imagenet':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform = TwoCropTransform(transform)
            
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
        return transform
            
            
    def train(self, vis_encoder, optimizer, device: str):
        total_loss = 0
        
        desc = '[TRAIN] Loss: {loss:>6.4f}'
        p_bar = tqdm(self.trainloader, desc=desc.format(loss=total_loss), leave=False)
        
        self.hflip.to(device)
        
        vis_encoder.train()
        
        for idx, (images, targets) in enumerate(p_bar):          
            # augment images
            if self.dataname != 'imagenet':
                bsz = images.size(0)
                images = images.to(device)
                images_pair = self.hflip(images.repeat(2, 1, 1, 1))  # 2B with hflip
            else:
                bsz = images[0].size(0)
                images1, images2 = images[0].to(device), images[1].to(device)
                images_pair = torch.cat([images1, images2], dim=0)  # 2B
            
            images_pair = self.simclr_aug(images_pair)  # simclr augment
            outputs = vis_encoder(images_pair, simclr=True, shift=True)
                       
            f1, f2 = torch.split(outputs['simclr'], [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            # loss
            loss = self.train_criterion(features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            p_bar.set_description(desc=desc.format(loss=total_loss/(idx+1)))
            
            
            
