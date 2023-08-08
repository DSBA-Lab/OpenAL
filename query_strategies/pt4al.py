import numpy as np
from torch.utils.data import Dataset, DataLoader

from .strategy import Strategy, SubsetSequentialSampler
import os
import pandas as pd

import torch


class PT4AL(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int, save_path: str, round: int, n_init: int):
        
        super(PT4AL, self).__init__(
            model       = model,
            n_query     = n_query, 
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
        self.cycle = 0
        self.save_path = save_path
        self.round = round
        self.n_init = n_init

            
    def extract_unlabeled_prob(self, model, n_subset: int = None) -> np.ndarray:
        
        self.cycle += 1
        
        # load ssl_pretext batch data
        ssl_batch_path = os.path.join(self.save_path, 'batch', f'batch_loss.txt')
        with open(ssl_batch_path, 'r') as f:
            samples = f.readlines()
        
        # pt4al batch indexing
        query_samples = samples[:len(samples)-self.n_init]
        query_samples_len = len(query_samples)
        query_samples[query_samples_len//self.round * (self.cycle-1) : query_samples_len//self.round * (self.cycle)]
            
        batch_idx = list(map(int, pd.DataFrame(query_samples)[0].str.replace('\n', '')))
        
        sampler = SubsetSequentialSampler(
            indices = self.subset_sampling(indices=batch_idx, n_subset=n_subset) if n_subset else batch_idx
        )
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        # predict probability on pt4al selected index
        probs = []
        
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                outputs = model(inputs.to(device))
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                probs.append(outputs.cpu())
                
        probs = torch.vstack(probs)
        
        return probs, batch_idx
    
          
    def query(self, model, n_subset: int = None) -> np.ndarray:
        
        probs, batch_idx = self.extract_unlabeled_prob(model, n_subset)
        
        # unlabeled index
        unlabeled_idx = np.array(batch_idx)
        
        # select least confidence
        max_confidence = probs.max(1)[0]
        select_idx = unlabeled_idx[max_confidence.sort()[1][:self.n_query]]
           
        return select_idx