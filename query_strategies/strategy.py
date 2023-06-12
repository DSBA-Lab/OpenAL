
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
        self, model, n_query: int, dataset: Dataset, labeled_idx: np.ndarray, batch_size: int, num_workers: int):
        
        self.model = model
        self.n_query = n_query
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
        probs = []
        
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                outputs = model(inputs.to(device))
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                probs.append(outputs.cpu())
                
        return torch.vstack(probs)

    def extract_unlabeled_prob_embed(self, model, n_subset: int = None) -> torch.Tensor:         
        
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
        probs = []
        embeds = []
        
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                outputs, embed = model.embed_forward(inputs.to(device))
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                probs.append(outputs.cpu())
                embeds.append(embed.cpu())

        return torch.vstack(probs), torch.vstack(embeds)
    
    def extract_labeled_prob_embed(self, model, n_subset: int = None) -> torch.Tensor:         
        
        # define sampler
        labeled_idx = np.where(self.labeled_idx==True)[0]
        sampler = SubsetSequentialSampler(
            indices = self.subset_sampling(indices=labeled_idx, n_subset=n_subset) if n_subset else labeled_idx
        )
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        # predict
        probs = []
        embeds = []
        Y = []
        
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i, (inputs, label) in enumerate(dataloader):
                outputs, embed = model.embed_forward(inputs.to(device))
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                probs.append(outputs.cpu())
                embeds.append(embed.cpu())
                Y.append(label.view(label.size(0),1).cpu())

        return torch.vstack(probs), torch.vstack(embeds), torch.vstack(Y)
