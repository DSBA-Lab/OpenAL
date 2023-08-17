import numpy as np
import torch
from torch.utils.data import Dataset

from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int, n_subset: int = 0):
        
        super(EntropySampling, self).__init__(
            model       = model,
            n_query     = n_query, 
            n_subset    = n_subset,
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
    def query(self, model) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
        
        # predict probability on unlabeled dataset
        probs = self.extract_outputs(
            model      = model, 
            sample_idx = unlabeled_idx, 
        )
        
        # select maximum entropy
        entropy = (-(probs*torch.log(probs))).sum(dim=1)
        select_idx = unlabeled_idx[entropy.sort(descending=True)[1][:self.n_query]]
        
        return select_idx