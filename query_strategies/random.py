import numpy as np

from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, **init_args):
        
        super(RandomSampling, self).__init__(**init_args)
        
    def query(self, model, **kwargs) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = kwargs.get('unlabeled_idx', self.get_unlabeled_idx())
        
        np.random.shuffle(unlabeled_idx)
        q_idx = self.query_interval(unlabeled_idx=unlabeled_idx, model=model)
        select_idx = unlabeled_idx[q_idx]
        
        return select_idx