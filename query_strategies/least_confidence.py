import numpy as np
from torch.utils.data import Dataset

from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int):
        
        super(LeastConfidence, self).__init__(
            model       = model,
            n_query     = n_query, 
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
    
    def query(self, model, n_subset: int = None) -> np.ndarray:
        
       # predict probability on unlabeled dataset
        probs = self.extract_unlabeled_prob(model=model, n_subset=n_subset)
        
        # unlabeled index
        unlabeled_idx = np.where(self.labeled_idx==False)[0]
        
        # select least confidence
        max_confidence = probs.max(1)[0]
        select_idx = unlabeled_idx[max_confidence.sort()[1][:self.n_query]]
        
        return select_idx