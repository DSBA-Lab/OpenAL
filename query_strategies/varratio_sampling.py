import numpy as np
from torch.utils.data import Dataset

from .strategy import Strategy

class VarRatioSampling(Strategy):
    '''
    VarRatioSampling : Sampling by variation and ratio
    Elementary applied statistics : for students in dehavioral science. New York: Wiley, 1965
    
    VarRatioSampling is the same as LeastConfidence
    '''
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int):
        
        super(VarRatioSampling, self).__init__(
            model       = model,
            n_query     = n_query, 
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
    def query(self, model) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
   
        # predict probability on unlabeled dataset
        probs = self.extract_unlabeled_prob(model=model, unlabeled_idx=unlabeled_idx)
        
        preds = probs.max(dim=1)[0]
        uncertainties = (1.0 - preds).sort(descending = True)[1]

        select_idx = unlabeled_idx[uncertainties[:self.n_query]]

        return select_idx