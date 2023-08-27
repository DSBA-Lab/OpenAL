import numpy as np
import torch
from torch.utils.data import Dataset
from .strategy import Strategy

class MeanSTDSampling(Strategy):
    '''
    Mean Standard Sampling (MeanSTD)
    '''
    def __init__(self, num_mcdropout: int = 10, **init_args):
        
        super(MeanSTDSampling, self).__init__(**init_args)
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