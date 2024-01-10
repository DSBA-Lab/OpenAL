import numpy as np
from torch.utils.data import Dataset

from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, **init_args):
        
        super(LeastConfidence, self).__init__(**init_args)
    
    def query(self, model, **kwargs) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = kwargs.get('unlabeled_idx', self.get_unlabeled_idx())
        
        # predict probability on unlabeled dataset
        probs = self.extract_outputs(
            model      = model, 
            sample_idx = unlabeled_idx, 
        )['probs']
        
        # select least confidence
        max_confidence = probs.max(1)[0]
        q_idx = self.query_interval(unlabeled_idx=unlabeled_idx, model=model)
        select_idx = unlabeled_idx[max_confidence.sort()[1][q_idx]]
        
        return select_idx