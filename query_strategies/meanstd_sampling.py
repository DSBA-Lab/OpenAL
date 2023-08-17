import numpy as np
import torch
from torch.utils.data import Dataset
from .strategy import Strategy

class MeanSTDSampling(Strategy):
    '''
    Mean Standard Sampling (MeanSTD)
    '''
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int, n_subset: int = 0, num_mcdropout: int = 10):
        
        super(MeanSTDSampling, self).__init__(
            model       = model,
            n_query     = n_query, 
            n_subset    = n_subset,
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )

        self.num_mcdropout = num_mcdropout
          
    def query(self, model) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
        
        # outputs: (num_mcdropout x samples x num_classes)
        outputs = self.extract_outputs(
            model         = model, 
            sample_idx    = unlabeled_idx, 
            num_mcdropout = self.num_mcdropout
        )
        
        # sigma_c: (samples x num_classes)
        sigma_c = torch.std(outputs, dim=0)
        
        # uncertainties: (samples, )
        uncertainties = torch.mean(sigma_c, dim=1)

        # select samples' index from largest uncertainty
        select_idx = unlabeled_idx[uncertainties.sort(descending=True)[1][:self.n_query]]

        return select_idx    