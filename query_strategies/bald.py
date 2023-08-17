import numpy as np
import torch
from torch.utils.data import Dataset

from .strategy import Strategy

class BALD(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int, n_subset: int = 0, num_mcdropout: int = 10):
        
        super(BALD, self).__init__(
            model       = model,
            n_query     = n_query, 
            n_subset    = n_subset,
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
        self.num_mcdropout = num_mcdropout
        
    
    def shannon_entropy_function(self, model, unlabeled_idx: np.ndarray):
        # outputs: (num_mcdropout x samples x num_classes)
        outputs = self.extract_outputs(
            model         = model, 
            sample_idx    = unlabeled_idx, 
            num_mcdropout = self.num_mcdropout
        )
        
        # pc: (samples x num_classes)
        pc = outputs.mean(dim=0)
        H = (-pc * torch.log(pc + 1e-10)).sum(dim=1)  # To avoid division with zero, add 1e-10
        E = -torch.mean(torch.sum(outputs * torch.log(outputs + 1e-10), dim=2), dim=0)
        return H, E
        
    def query(self, model) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
        
        # predict probability on unlabeled dataset
        H, E_H = self.shannon_entropy_function(model=model, unlabeled_idx=unlabeled_idx)
        
        # calculate mutual information 
        mutual_information = H - E_H 
        
        # select maximum mutual_information
        select_idx = unlabeled_idx[mutual_information.sort(descending=True)[1][:self.n_query]]
        
        return select_idx
    
    