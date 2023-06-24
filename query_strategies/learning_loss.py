import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from functools import partial
from collections import OrderedDict
from .strategy import Strategy, SubsetSequentialSampler


class LearningLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(LearningLoss, self).__init__()
        
        self.margin = margin
    
    def forward(self, outputs, targets):
        '''
        batch size should be even
        but we can't always be satisfied, so we cut our losses and exclude the last sample if it's an odd number.
        '''
        if len(outputs) % 2 != 0:
            outputs = outputs[:-1]
            targets = targets[:-1]
        
        indicate = torch.where((targets[::2]-targets[1::2])>0, 1, -1)
        loss = torch.clamp(-indicate * (outputs[::2]-outputs[1::2]) + self.margin, min=0)
        
        return loss
        
    
class LossPredictionModule(nn.Module):
    def __init__(self, layer_ids: list, in_features_list: list, out_features: int = 128, channel_last: bool = False):
        super(LossPredictionModule, self).__init__()
        self.layer_ids = layer_ids
        self.in_features_list = in_features_list
        self.out_features = out_features
        self.channel_last = channel_last
        
        # make branch per layer
        for i, layer_id in enumerate(self.layer_ids):
            setattr(self, layer_id, self.make_branch(in_features=self.in_features_list[i]))
        
        # last fully connnected layer
        self.fc = nn.Linear(out_features*len(in_features_list), 1)
        
        
    def make_branch(self, in_features: int) -> nn.Sequential:
        return nn.Sequential(OrderedDict([
            ('gap', nn.AdaptiveAvgPool2d((1,1))),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(in_features, self.out_features)),
            ('relu', nn.ReLU())
        ]))
        
        
    def forward(self, x: dict):
        
        out_features = []
        for layer_id in self.layer_ids:
            x_i = x[layer_id]
            if self.channel_last:
                x_i = x_i.permute(0,3,1,2)
            out_features.append(getattr(self, layer_id)(x_i))

        out_features = torch.cat(out_features, dim=1)
        out_loss = self.fc(out_features).view(-1)
        
        return out_loss
        
    
    
class LearningLossModel(nn.Module):
    def __init__(
        self, backbone, layer_ids: list, in_features_list: list, out_features: int = 128, 
        in_layer: bool = False, channel_last: bool = False):
        super(LearningLossModel, self).__init__()
        
        self.backbone = backbone        
        self.layer_ids = layer_ids
        self.in_layer = in_layer
        self.layer_outputs = {}
        self.save_forward_output()
        
        self.LPM = LossPredictionModule(
            layer_ids        = self.lpm_layer_ids, 
            in_features_list = in_features_list, 
            out_features     = out_features,
            channel_last     = channel_last
        )
    
    def save_forward_output(self):
        def hook_forward(module, input, output, key):
            self.layer_outputs[key] = output
        
        lpm_layer_ids = []
        for layer_id in self.layer_ids:
            if not self.in_layer:
                self.backbone._modules[layer_id].register_forward_hook(partial(hook_forward, key=layer_id))
            else:
                for idx, in_module in enumerate(self.backbone._modules[layer_id]):
                    in_module.register_forward_hook(partial(hook_forward, key=f'{layer_id}{idx}'))
                    lpm_layer_ids.append(f'{layer_id}{idx}')
        
        if self.in_layer:
            setattr(self, 'lpm_layer_ids', lpm_layer_ids)
    
    def forward(self, x):
        out_y = self.backbone(x)
        out_loss = self.LPM(self.layer_outputs)
        
        return {
            'logits'    : out_y, 
            'loss_pred' : out_loss
        }
    
    
    
class LearningLossAL(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int, 
        margin: float, loss_weight: float, layer_ids: list, in_features_list: list, out_features: int = 128, 
        in_layer: bool = False, channel_last: bool = False):
        
        model = LearningLossModel(
            backbone         = model, 
            layer_ids        = layer_ids, 
            in_features_list = in_features_list, 
            out_features     = out_features,
            in_layer         = in_layer,
            channel_last     = channel_last
        )
        
        super(LearningLossAL, self).__init__(
            model       = model,
            n_query     = n_query, 
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
        self.criterion.reduction = 'none'
        self.learningloss = LearningLoss(margin=margin)
        self.loss_weight = loss_weight
    
    
    def query(self, model, n_subset: int = None) -> np.ndarray:
        
       # predict loss-prediction on unlabeled dataset
        loss_pred = self.extract_unlabeled_prob(model=model, n_subset=n_subset)
        
        # unlabeled index
        unlabeled_idx = np.where(self.labeled_idx==False)[0]
        
        # select loss
        select_idx = unlabeled_idx[loss_pred.sort(descending=True)[1][:self.n_query]]
        
        return select_idx
    

    def init_model(self):
        model = deepcopy(self.model)
        model.save_forward_output()
        
        return model

        
    def loss_fn(self, outputs, targets):
        target_loss = self.criterion(outputs['logits'], targets)
        loss_pred_loss = self.learningloss(outputs['loss_pred'], target_loss)
    
        return target_loss.mean() + (self.loss_weight * loss_pred_loss.mean())
    
    
    def extract_unlabeled_prob(self, model, n_subset: int = None) -> torch.Tensor:         
        
        # define sampler
        unlabeled_idx = np.where(self.labeled_idx==False)[0]
        sampler = SubsetSequentialSampler(
            indices = self.subset_sampling(indices=unlabeled_idx, n_subset=n_subset) if n_subset else unlabeled_idx
        )
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        # predict
        loss_pred = []
        
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                outputs = model(inputs.to(device))
                loss_pred.append(outputs['loss_pred'].cpu())
                
        return torch.cat(loss_pred)