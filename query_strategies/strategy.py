
import numpy as np
import torch
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

    def extract_unlabeled_prob(self, model, unlabeled_idx: np.ndarray) -> torch.Tensor:         
        # define sampler
        sampler = SubsetSequentialSampler(indices=unlabeled_idx)
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        # predict
        probs = []
        
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                outputs = model(inputs.to(device))
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                probs.append(outputs.cpu())
                
        return torch.vstack(probs)