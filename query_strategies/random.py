import numpy as np
from torch.utils.data import Dataset

from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int):
        
        super(RandomSampling, self).__init__(
            model       = model,
            n_query     = n_query, 
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
    def query(self, model, n_subset: int = None) -> np.ndarray:
        
        # unlabeled index
        unlabeled_idx = np.where(self.labeled_idx==False)[0]
        
        np.random.shuffle(unlabeled_idx)
        select_idx = unlabeled_idx[:self.n_query]
        
        return select_idx