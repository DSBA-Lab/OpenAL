import numpy as np
import torch

from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, **init_args):
        
        super(EntropySampling, self).__init__(**init_args)
    
    def get_scores(self, model, sample_idx: np.ndarray):
        
        # predict probability on unlabeled dataset
        probs = self.extract_outputs(
            model      = model, 
            sample_idx = sample_idx, 
        )['probs']
        
        # select maximum entropy
        entropy = (-(probs*torch.log(probs))).sum(dim=1)
        
        return entropy.sort(descending=True)[1]
        