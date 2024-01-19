import numpy as np
from torch.utils.data import Dataset

from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, **init_args):
        
        super(LeastConfidence, self).__init__(**init_args)
    
    def get_scores(self, model, sample_idx: np.ndarray):
        # predict probability on unlabeled dataset
        probs = self.extract_outputs(
            model      = model, 
            sample_idx = sample_idx, 
        )['probs']
        
        max_confidence = 1 - probs.max(dim=1)[0]
        
        return max_confidence, max_confidence.sort(descending=True)[1]