import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import Dataset, DataLoader
from .strategy import Strategy,SubsetSequentialSampler

class BALD(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int, n_subset: int = 0, num_mcdropout: int = 10):
        
        super(BALD, self).__init__(
            model       = model,
            n_query     = n_query, 
            n_subset    = n_subset,
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
        self.num_mcdropout = num_mcdropout
        
        
    def extarct_unlabeled_prob(self, model, unlabeled_idx: np.ndarray) -> torch.Tensor:
        # define sampler 
        sampler = SubsetSequentialSampler(indices=unlabeled_idx)
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        # predict
        device = next(model.parameters()).device
        model.train()
        with torch.no_grad():
            probs = [] 
            # iteration for the number of MC Dropout
            for i in range(self.num_mcdropout):
                mc_probs = [] 
                
                for j, (inputs,_) in enumerate(dataloader):
                    outputs = model(inputs.to(device))
                    outputs = torch.nn.functional.softmax(outputs,dim=1)
                    mc_probs.extend(outputs.detach().cpu().numpy())
                probs.append(mc_probs)
        probs = np.array(probs)        
        
        return probs 
    
    def shannon_entropy_function(self, model, unlabeled_idx: np.ndarray):
        outputs = self.extarct_unlabeled_prob(model=model, unlabeled_idx=unlabeled_idx)
        pc = outputs.mean(axis=0)
        H = (-pc * np.log(pc + 1e-10)).sum(axis=-1)  # To avoid division with zero, add 1e-10
        E = -np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
        return H, E
        
    def query(self, model) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
        
        # predict probability on unlabeled dataset
        H, E_H = self.shannon_entropy_function(model=model, unlabeled_idx=unlabeled_idx)
        
        # calculate mutual information 
        mutual_information = H - E_H 
        
        # select maximum mutual_information
        select_idx = unlabeled_idx[torch.Tensor(mutual_information).sort(descending=True)[1][:self.n_query]]
        
        return select_idx
    
    