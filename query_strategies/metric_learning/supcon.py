import numpy as np
import torch
import torch.nn as nn
import random

from PIL import Image
from copy import deepcopy
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from .factory import MetricLearning
from .losses import SupConLoss
from .transform_layers import TwoCropTransform, get_simclr_aug


class SupCon(MetricLearning):
    def __init__(self, train_transform, batch_size: int, num_workers: int, **init_params):
        super(SupCon, self).__init__(**init_params)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.criterion = SupConLoss()
    
    def create_trainset(self, dataset, sample_idx: np.ndarray, **kwargs):
        # set trainset        
        trainset = deepcopy(dataset)
        trainset.transform = TwoCropTransform(
            get_simclr_aug(
                img_size  = self.train_transform.transforms[0].size, 
                aug_info  = [
                    'RandomResizedCrop',
                    'RandomHorizontalFlip',
                    'ColorJitter',
                    'RandomGrayscale',
                    'ToTensor'
                ],
                normalize = self.train_transform.transforms[-1]
        ))
        trainloader = DataLoader(trainset, sampler=SubsetRandomSampler(indices=sample_idx), batch_size=self.batch_size, num_workers=self.num_workers)

        # set attributions for trainloader and testset
        setattr(self, 'trainloader', trainloader)
        
            
    def train(self, vis_encoder, optimizer, device: str, **kwargs):
        total_loss = 0
        
        desc = '[TRAIN] Loss: {loss:>6.4f}, Logit scaler: {scaler:>6.4f}'
        p_bar = tqdm(self.trainloader, desc=desc.format(loss=total_loss, scaler=0.0), leave=False)
        
        vis_encoder.train()
        
        for idx, (images, targets) in enumerate(p_bar):
            bsz = targets.size(0)
            targets = targets.to(device)
            images = torch.cat([images[0], images[1]], dim=0).to(device)

            features, logit_scale = vis_encoder(images, return_logits_scaler=True)
            
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            loss = self.criterion(features=features, temperature=1/logit_scale, labels=targets)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            p_bar.set_description(desc=desc.format(loss=total_loss/(idx+1), scaler=logit_scale))
    
    
class SupCon2(MetricLearning):
    def __init__(self, train_transform, **init_params):
        
        super(SupCon2, self).__init__(**init_params)   
        
        self.train_transform = train_transform
        self.criterion = nn.CrossEntropyLoss()
    
    def create_trainset(self, dataset, sample_idx: np.ndarray, **kwargs):
        # set trainset        
        _, train_cls_cnt = np.unique(dataset.targets[sample_idx], return_counts=True)
        train_num_batch = min(train_cls_cnt)

        trainset = SupConDataset(
            num_batchs = train_num_batch,
            data       = dataset.data[sample_idx], 
            labels     = dataset.targets[sample_idx], 
            transform  = self.train_transform
        )
        
        # set attributions for trainset and testset
        setattr(self, 'trainset', trainset)
    
    def train(self, vis_encoder, optimizer, device: str, **kwargs):
        total_loss = 0
        
        desc = '[TRAIN] Loss: {loss:>6.4f}, Logit scaler: {scaler:>6.4f}'
        p_bar = tqdm(self.trainset, desc=desc.format(loss=total_loss, scaler=0.0), leave=False)
        
        vis_encoder.train()
        
        for idx, (anchor, positive) in enumerate(p_bar):
            if idx == len(self.trainset):
                break

            anchor, positive = anchor.to(device), positive.to(device)

            out_anchor, logit_scale = vis_encoder(anchor, return_logits_scaler=True)
            out_positive = vis_encoder(positive)

            targets_i = torch.arange(anchor.size(0)).to(device)

            similarity = torch.einsum('ae, pe -> ap', out_anchor, out_positive)
            similarity = similarity * logit_scale
            loss1 = self.criterion(similarity, targets_i)
            loss2 = self.criterion(similarity.t(), targets_i)
            loss = (loss1 + loss2) / 2
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            p_bar.set_description(desc=desc.format(loss=total_loss/(idx+1), scaler=logit_scale))
    
    
class SupConDataset(Dataset):
    def __init__(self, num_batchs, data, labels, transform=None):
        self.num_batchs = num_batchs
        self.data = data
        self.labels = labels
        self.transform = transform
        self.num_classes = len(np.unique(labels))
        self.class_idx = dict([(c, np.where(labels==c)[0]) for c in range(self.num_classes)])
    
    def __len__(self):
        return self.num_batchs
    
    def __getitem__(self, i):
        anchors = []
        positives = []
        
        for c in range(self.num_classes):
            # examples in class y_i
            idx_c = self.class_idx[c]
        
            # random choice anchor index and positive index for anchor
            anchor_idx = random.choice(idx_c)
            positive_idx = random.choice(idx_c)
        
            while positive_idx == anchor_idx:
                positive_idx = random.choice(idx_c)
            
            # select anchor and positive image for anchor
            anchor = self.data[anchor_idx]
            positive = self.data[positive_idx]
            
            if self.transform != None:
                anchor = self.transform(Image.fromarray(anchor))
                positive = self.transform(Image.fromarray(positive))
                
            anchors.append(anchor)
            positives.append(positive)
            
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)

        return [anchors, positives]
    