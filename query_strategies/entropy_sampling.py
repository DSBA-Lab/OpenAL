import numpy as np
import torch

from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, **init_args):
        
        super(EntropySampling, self).__init__(**init_args)
        
    def query(self, model, **kwargs) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = kwargs.get('unlabeled_idx', self.get_unlabeled_idx())
        
        # predict probability on unlabeled dataset
        probs = self.extract_outputs(
            model      = model, 
            sample_idx = unlabeled_idx, 
        )['probs']
        
        # select maximum entropy
        entropy = (-(probs*torch.log(probs))).sum(dim=1)
        q_idx = self.query_interval(unlabeled_idx=unlabeled_idx, model=model)
        select_idx = unlabeled_idx[entropy.sort(descending=True)[1][q_idx]]
        
        return select_idx