import numpy as np
from torch.utils.data import Dataset

from .strategy import Strategy

class VarRatioSampling(Strategy):
    '''
    VarRatioSampling : Sampling by variation and ratio
    Elementary applied statistics : for students in dehavioral science. New York: Wiley, 1965
    
    VarRatioSampling is the same as LeastConfidence
    '''
    def __init__(self, **init_args):
        
        super(VarRatioSampling, self).__init__(**init_args)
        
    def query(self, model) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
   
        # predict probability on unlabeled dataset
        probs = self.extract_outputs(
            model      = model, 
            sample_idx = unlabeled_idx, 
        )['probs']
        
        preds = probs.max(dim=1)[0]
        uncertainties = (1.0 - preds).sort(descending = True)[1]

        select_idx = unlabeled_idx[uncertainties[:self.n_query]]

        return select_idx