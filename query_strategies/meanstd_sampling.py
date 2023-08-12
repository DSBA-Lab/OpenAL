import numpy as np
import torch
from torch.utils.data import Dataset
from .strategy import Strategy

from torch.utils.data import Dataset, DataLoader
from .strategy import Strategy, SubsetSequentialSampler

class MeanSTDSampling(Strategy):
    '''
    Mean Standard Sampling (MeanSTD)
    '''
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int, num_mcdropout: int = 10):
        
        super(MeanSTDSampling, self).__init__(
            model       = model,
            n_query     = n_query, 
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
          
    def query(self, model) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
        
        # 라벨링되지 않은 데이터셋에 대한 확률값 예측
        outputs = self.extarct_unlabeled_prob(model=model, unlabeled_idx=unlabeled_idx)
        sigma_c = np.std(outputs, axis=0)
        uncertainties = np.mean(sigma_c, axis=-1)

        select_idx = unlabeled_idx[torch.Tensor(-np.array(uncertainties)).sort(descending=True)[1][:self.n_query]]

        return select_idx    