import numpy as np
import torch
from .strategy import Strategy

class MeanSTDSampling(Strategy):
    '''
    Mean Standard Sampling (MeanSTD)
    '''
    def __init__(self, num_mcdropout: int = 10, **init_args):
        
        super(MeanSTDSampling, self).__init__(**init_args)
        self.num_mcdropout = num_mcdropout
          
    
    def get_scores(self, model, sample_idx: np.ndarray):
        # outputs: (num_mcdropout x samples x num_classes)
        outputs = self.extract_outputs(
            model         = model, 
            sample_idx    = sample_idx, 
            num_mcdropout = self.num_mcdropout
        )
        
        # sigma_c: (samples x num_classes)
        sigma_c = torch.std(outputs, dim=0)
        
        # uncertainties: (samples, )
        uncertainties = torch.mean(sigma_c, dim=1)
        
        return uncertainties.sort(descending=True)[1]