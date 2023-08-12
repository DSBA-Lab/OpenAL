import numpy as np
from torch.utils.data import Dataset

from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int, n_subset: int = 0):
        
        super(MarginSampling, self).__init__(
            model       = model,
            n_query     = n_query, 
            n_subset    = n_subset,
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
        
        # select margin between top two class probability
        sorted_desc_prob, _ = probs.sort(descending=True)
        prob_margin = sorted_desc_prob[:,0] - sorted_desc_prob[:,1]
        select_idx = unlabeled_idx[(prob_margin).sort()[1][:self.n_query]]
        
        return select_idx