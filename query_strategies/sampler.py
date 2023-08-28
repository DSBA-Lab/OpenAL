import numpy as np
import torch
from torch.utils.data import Sampler

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices: np.ndarray):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    

class SubsetWeightedRandomSampler(Sampler):
    def __init__(self, indices: np.ndarray, weights: list, replacement: bool = True):
        self.indices = indices
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        rand_tensor = torch.multinomial(
            input       = self.weights, 
            num_samples = len(self.indices), 
            replacement = self.replacement
        )
        
        return (self.indices[i] for i in rand_tensor)
    
    def __len__(self):
        return len(self.indices)