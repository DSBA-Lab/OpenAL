import numpy as np
from torch.utils.data import Dataset

from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, **init_args):
        
        super(MarginSampling, self).__init__(**init_args)
    
    def get_scores(self, model, sample_idx: np.ndarray):
        # predict probability on unlabeled dataset
        probs = self.extract_outputs(
            model      = model, 
            sample_idx = sample_idx, 
        )['probs']
        
        sorted_desc_prob, _ = probs.sort(descending=True)
        prob_margin = sorted_desc_prob[:,0] - sorted_desc_prob[:,1]
        
        return prob_margin.sort()[1]