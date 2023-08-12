import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from query_strategies import LeastConfidence
from .make_startset import get_batch_params

class PT4AL:
    def __init__(
        self, n_query: int, n_start: int, n_end: int, batch_path: str):
        
        # round log
        self.r = 1
        
        # batch file
        self.batch_path = batch_path
        self.batch_idx = pd.read_csv(batch_path)['idx'].values
        
        # al setting
        self.n_start = n_start
        self.n_end = n_end
        self.total_round, self.b_size, self.b_init, self.sampling_interval = get_batch_params(
            batch_size = len(self.batch_idx),
            n_start    = n_start,
            n_end      = n_end,
            n_query    = n_query
        )
        
    def batch_sampling(self, r: int):
        # current used batch size
        current_size = self.b_init if r == 1 else self.b_init + (self.b_size * (r-1))
        
        # select index for query sampling
        selected_idx = range(current_size, current_size + self.b_size, self.sampling_interval)[:self.n_query]
        
        return self.batch_idx[selected_idx]

            
    def get_unlabeled_idx(self):
        unlabeled_idx = self.batch_sampling(r=self.r)
        self.r += 1
        
        return unlabeled_idx
    
    
class PT4LeastConfidence(PT4AL, LeastConfidence):
    def __init__(self, model, n_query: int, labeled_idx: np.ndarray, dataset: Dataset, batch_size: int, num_workers: int, 
        n_start: int, n_end: int, batch_path: str, n_subset: int = 0):
        
        super(PT4AL, self).__init__(
            n_query    = n_query, 
            n_start    = n_start, 
            n_end      = n_end, 
            batch_path = batch_path
        )
        
        super(LeastConfidence, self).__init__(
            model       = model,
            n_query     = n_query, 
            n_subset    = n_subset,
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        