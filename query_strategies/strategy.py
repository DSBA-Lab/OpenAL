
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

class Strategy:
    def __init__(
        self, n_query: int, dataset: Dataset, labeled_idx: np.ndarray, batch_size: int, num_workers: int):
        
        self.n_query = n_query
        self.labeled_idx = labeled_idx 
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def loss_fn(self, outputs, targets):
        return self.criterion(outputs, targets)
        
    def query(self):
        raise NotImplementedError
    
    def update(self, query_idx: np.ndarray) -> DataLoader:
        
        self.labeled_idx[query_idx] = True
        
        dataloader = DataLoader(
            dataset     = self.dataset_sampling(sample_idx=self.labeled_idx),
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers
        )
        
        return dataloader
    
    def dataset_sampling(self, sample_idx: np.ndarray) -> Dataset:
        # define unlabeled dataset
        sampled_dataset = deepcopy(self.dataset)
        
        if sampled_dataset.__class__.__name__ == 'ALDataset':
            sampled_dataset.data_info = sampled_dataset.label_info[sample_idx]
        else:
            sampled_dataset.data = sampled_dataset.data[sample_idx]
            sampled_dataset.targets = np.array(sampled_dataset.targets)[sample_idx]

        return sampled_dataset


    def extract_unlabeled_prob(self, model) -> torch.Tensor:         
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = self.dataset_sampling(sample_idx=~self.labeled_idx),
            batch_size  = self.batch_size,
            shuffle     = False,
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
