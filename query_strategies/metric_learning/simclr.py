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
from .losses import SupConLoss, nt_xent_loss
from .transform_layers import get_simclr_aug, get_simclr_augmentation, TwoCropTransform, HorizontalFlipLayer, Rotation, CutPerm


class SimCLRCSI(MetricLearning):
    def __init__(self, dataname: str, img_size: int, batch_size: int, num_workers: int, shift_trans_type: str = 'rotation', sim_lambda: float = 1.0, **init_params):
        super(SimCLRCSI, self).__init__(**init_params)
        
        self.sim_lambda = sim_lambda
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.temperature = 0.5
        self.criterion = SupConLoss()
        self.clf_criterion = nn.CrossEntropyLoss()
        
        # get shift transform
        self.get_shift_module(shift_trans_type=shift_trans_type)
        
        self.dataname = dataname
        
        # simclr_aug
        aug_info = ['ColorJitter', 'RandomGrayscale', 'RandomResizedCrop']
        aug_info = aug_info[:-1] if dataname == 'imagenet' else aug_info
        
        # self.simclr_aug = get_simclr_aug(img_size=img_size, aug_info=aug_info)
        self.simclr_aug = get_simclr_augmentation(img_size=img_size)
        
        # hflip
        self.hflip = HorizontalFlipLayer()
    
    def create_trainset(self, dataset, sample_idx: np.ndarray, **kwargs):
        # set trainset        
        trainset = deepcopy(dataset)
        # trainset.transform = self.create_transform()
        trainloader = DataLoader(trainset, sampler=SubsetRandomSampler(indices=sample_idx), batch_size=self.batch_size, num_workers=self.num_workers)
        
        # set attributions for trainloader and testset
        setattr(self, 'trainloader', trainloader)
        
    def create_transform(self, test: bool = False):
        if self.dataname == 'imagenet':
            if test:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
            else:
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
            
    def train(self, epoch, vis_encoder, optimizer, scheduler, device: str, **kwargs):
        total_sim_loss = 0
        total_shift_loss = 0
        
        desc = '[TRAIN] LR: {lr:.3e} Sim Loss: {sim_loss:>6.4f} Shift Loss: {shift_loss:>6.4f}'
        p_bar = tqdm(self.trainloader, desc=desc.format(lr=optimizer.param_groups[0]['lr'], sim_loss=0, shift_loss=0), leave=False)
        
        self.hflip.to(device)
        self.simclr_aug.to(device)
        
        vis_encoder.train()
        
        for idx, (images, targets) in enumerate(p_bar):          
            # augment images
            if self.dataname != 'imagenet':
                images = images.to(device)
                images1, images2 = self.hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
            else:
                images1, images2 = images[0].to(device), images[1].to(device)
            
            images1 = torch.cat([self.shift_transform(images1, k) for k in range(self.k_shift)])
            images2 = torch.cat([self.shift_transform(images2, k) for k in range(self.k_shift)])
            shift_labels = torch.cat([torch.ones_like(targets) * k for k in range(self.k_shift)], 0)  # B -> 4B
            shift_labels = shift_labels.repeat(2).to(device)
            
            images_pair = torch.cat([images1, images2], dim=0)
            images_pair = self.simclr_aug(images_pair)  # simclr augment
            outputs = vis_encoder(images_pair, shift=True)
            
            # features = torch.stack(outputs['simclr'].chunk(2), dim=1)
            features = torch.matmul(outputs['simclr'], outputs['simclr'].t())
            
            # loss
            loss_sim = nt_xent_loss(features, temperature=self.temperature)
            
            loss_shift = self.clf_criterion(outputs['shift'], shift_labels)
            loss = loss_sim * self.sim_lambda + loss_shift
            total_sim_loss += loss_sim.item()
            total_shift_loss += loss_shift.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step(epoch - 1 + idx / len(self.trainloader))
            
            p_bar.set_description(desc=desc.format(lr=optimizer.param_groups[0]['lr'], sim_loss=total_sim_loss/(idx+1), shift_loss=total_shift_loss/(idx+1)))
            
            
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
        
        setattr(self, 'shift_transform', shift_transform)
        setattr(self, 'k_shift', k_shift)
            
            
    
class SimCLR(MetricLearning):
    def __init__(self, dataname: str, img_size: int, batch_size: int, num_workers: int, **init_params):
        super(SimCLRCSI, self).__init__(**init_params)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.criterion = SupConLoss()
        self.temperature = 0.5
        
        self.dataname = dataname
        
        # simclr_aug
        aug_info = ['ColorJitter', 'RandomGrayscale', 'RandomResizedCrop']
        aug_info = aug_info[:-1] if dataname == 'imagenet' else aug_info
        
        self.simclr_aug = get_simclr_aug(img_size=img_size, aug_info=aug_info)
        
        # hflip
        self.hflip = HorizontalFlipLayer()
        
    def create_trainset(self, dataset, sample_idx: np.ndarray, **kwargs):
        
        # set trainset        
        trainset = deepcopy(dataset)
        trainset.transform = self.create_transform()
        trainloader = DataLoader(trainset, sampler=SubsetRandomSampler(indices=sample_idx), batch_size=self.batch_size, num_workers=self.num_workers)
        
        # set attributions for trainloader and testset
        setattr(self, 'trainloader', trainloader)
        
        
    def create_transform(self, test: bool = False):
        if self.dataname == 'imagenet':
            if test:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
            else:
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
            
            
    def train(self, epoch, vis_encoder, optimizer, scheduler, device: str, **kwargs):
        total_loss = 0
        
        desc = '[TRAIN] LR: {lr:.3e} Loss: {loss:>6.4f}'
        p_bar = tqdm(self.trainloader, desc=desc.format(lr=optimizer.param_groups[0]['lr'], loss=total_loss), leave=False)
        
        self.hflip.to(device)
        
        vis_encoder.train()
        
        for idx, (images, targets) in enumerate(p_bar):          
            # augment images
            if self.dataname != 'imagenet':
                images = images.to(device)
                images_pair = self.hflip(images.repeat(2, 1, 1, 1))  # 2B with hflip
            else:
                images1, images2 = images[0].to(device), images[1].to(device)
                images_pair = torch.cat([images1, images2], dim=0)  # 2B
            
            images_pair = self.simclr_aug(images_pair)  # simclr augment
            outputs = vis_encoder(images_pair)
                       
            features = torch.stack(outputs['simclr'].chunk(2), dim=1)
            
            # loss
            loss = self.criterion(features, temperature=self.temperature)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step(epoch - 1 + idx / len(self.trainloader))
            
            p_bar.set_description(desc=desc.format(lr=optimizer.param_groups[0]['lr'], loss=total_loss/(idx+1)))
            
            
            
