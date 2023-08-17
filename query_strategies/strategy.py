
import numpy as np
import torch
from collections import defaultdict
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices: np.ndarray):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    

class Strategy:
    def __init__(
        self, model, n_query: int, dataset: Dataset, labeled_idx: np.ndarray, 
        batch_size: int, num_workers: int, n_subset: int = 0):
        
        self.model = model
        self.n_query = n_query
        self.n_subset = n_subset
        self.labeled_idx = labeled_idx 
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def init_model(self):
        return deepcopy(self.model)
        
    def loss_fn(self, outputs, targets):
        return self.criterion(outputs, targets)
        
    def query(self):
        raise NotImplementedError
    
    def update(self, query_idx: np.ndarray) -> DataLoader:
        
        self.labeled_idx[query_idx] = True
        
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = SubsetRandomSampler(indices=np.where(self.labeled_idx==True)[0]),
            num_workers = self.num_workers
        )
        
        return dataloader

    def subset_sampling(self, indices: np.ndarray, n_subset: int):
        # define subset
        subset_indices = np.random.choice(indices, size=n_subset, replace=False)
            
        return subset_indices

    def get_unlabeled_idx(self):
        unlabeled_idx = np.where(self.labeled_idx==False)[0]
        if self.n_subset > 0:
            assert self.n_subset > self.n_query, 'the number of subset must larger than the number of query.'
            unlabeled_idx = self.subset_sampling(indices=unlabeled_idx, n_subset=self.n_subset)
            
        return unlabeled_idx

    def extract_outputs(self, model, sample_idx: np.ndarray, num_mcdropout: int = 0,
                        return_probs: bool = True, return_embed: bool = False, return_labels: bool = False) -> torch.Tensor or dict:
        
        # define sampler
        sampler = SubsetSequentialSampler(indices=sample_idx)
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        # inference
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            if num_mcdropout > 0:
                # results type is torch.Tensor
                results = self.mcdrop_outputs(
                    model         = model, 
                    dataloader    = dataloader, 
                    device        = device, 
                    num_mcdropout = num_mcdropout
                )
            else:
                # results type is dict
                results = self.get_outputs(
                    model         = model,
                    dataloader    = dataloader,
                    device        = device,
                    return_probs  = return_probs,
                    return_embed  = return_embed,
                    return_labels = return_labels
                )
                
        return results
    
    
    def get_outputs(
        self, model, dataloader, device: str, 
        return_probs: bool = True, return_embed: bool = False, return_labels: bool = False) -> dict:
        
        # predict
        results = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    # for labeled dataset that contains labels
                    inputs, labels = batch
                else:
                    # for unlabeled dataset that does not contain labels
                    inputs = batch
                
                # return labels    
                if return_labels:
                    results['labels'].append(labels.cpu())
                    
                # return embedings            
                if return_embed:
                    embed = model.forward_features(inputs.to(device))
                    results['embed'].append(embed.cpu())
                    forward_func = 'forward_head'
                else:
                    embed = inputs
                    forward_func = 'forward'
                
                # return probs
                if return_probs:
                    # forward head
                    outputs = model.__getattribute__(forward_func)(embed.to(device))
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                    results['probs'].append(outputs.cpu())
                
        # stack
        for k, v in results.items():
            if k == 'labels':
                results[k] = torch.hstack(v)
            else:
                results[k] = torch.vstack(v)
    
        return results
    
    
    def mcdrop_outputs(self, model, dataloader, device: str, num_mcdropout: int) -> torch.Tensor:
        # predict
        model.train()
        
        mc_probs = []
        # iteration for the number of MC Dropout
        for _ in range(num_mcdropout):
            probs = self.get_outputs(model=model, dataloader=dataloader, device=device)['probs']
            mc_probs.append(probs)
            
        return torch.stack(mc_probs)