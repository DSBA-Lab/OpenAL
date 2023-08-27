import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from query_strategies import LeastConfidence, MarginSampling, EntropySampling
from .make_startset import get_batch_params

class PT4AL:
    def __init__(self, n_query: int, n_start: int, n_end: int, batch_path: str):
        
        # round log
        self.r = 1
        
        # batch file
        self.batch_path = batch_path
        self.batch_idx = pd.read_csv(batch_path)['idx'].values
        
        # al setting
        self.n_start = n_start
        self.n_end = n_end
        _, self.b_size, self.b_init, _ = get_batch_params(
            batch_size = len(self.batch_idx),
            n_start    = n_start,
            n_end      = n_end,
            n_query    = n_query
        )
        
    def batch_sampling(self, r: int):
        # current used batch size
        current_size = self.b_init if r == 1 else self.b_init + (self.b_size * (r-1))
        
        # select index for query sampling
        selected_idx = range(current_size, current_size + self.b_size)
        
        return self.batch_idx[selected_idx]
    
    def get_unlabeled_idx(self):
        unlabeled_idx = self.batch_sampling(r=self.r)
        self.r += 1
        
        return unlabeled_idx
    
    
class PT4LeastConfidence(PT4AL, LeastConfidence):
    def __init__(self, batch_path: str, **init_args):
        
        super(PT4LeastConfidence, self).__init__(
            n_query    = init_args['n_query'], 
            n_start    = init_args['n_start'], 
            n_end      = init_args['n_end'], 
            batch_path = batch_path
        )
        
        super(LeastConfidence, self).__init__(**init_args)
        
    
    
class PT4MarginSampling(PT4AL, MarginSampling):
    def __init__(self, batch_path: str, **init_args):
        
        super(PT4MarginSampling, self).__init__(
            n_query    = init_args['n_query'], 
            n_start    = init_args['n_start'], 
            n_end      = init_args['n_end'], 
            batch_path = batch_path
        )
        
        super(MarginSampling, self).__init__(**init_args)
        

class PT4EntropySampling(PT4AL, EntropySampling):
    def __init__(self, batch_path: str, **init_args):
        
        super(PT4EntropySampling, self).__init__(
            n_query    = init_args['n_query'], 
            n_start    = init_args['n_start'], 
            n_end      = init_args['n_end'], 
            batch_path = batch_path
        )
        
        super(EntropySampling, self).__init__(**init_args)