import numpy as np
import torch
from torch.utils.data import Dataset
from .strategy import Strategy

from torch.utils.data import Dataset, DataLoader
from .strategy import Strategy,SubsetSequentialSampler

# Mean Standard Sampling 
class MeanSTDSampling(Strategy):
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

    def extarct_unlabeled_prob(self, model, n_subset: int = None) -> torch.Tensor:
        # define sampler
        unlabeled_idx = np.where(self.labeled_idx==False)[0]        
        sampler = SubsetSequentialSampler(
            indices = self.subset_sampling(indices=unlabeled_idx, n_subset=n_subset) if n_subset else unlabeled_idx
        )
        
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
            for i in range(self.num_mcdropout):
                mc_probs = [] 
                for j, (inputs,_) in enumerate(dataloader):
                    outputs = model(inputs.to(device))
                    outputs = torch.nn.functional.softmax(outputs,dim=1)
                    mc_probs.extend(outputs.detach().cpu().numpy())
                probs.append(mc_probs)
        probs = np.array(probs)        
        return probs, unlabeled_idx   
          
    def query(self, model, n_subset: int = None) -> np.ndarray:
        
        # 라벨링되지 않은 데이터셋에 대한 확률값 예측
        outputs,random_subset = self.extarct_unlabeled_prob(model, n_subset)
        sigma_c = np.std(outputs, axis = 0)
        _uncertainties = np.mean(sigma_c, axis = -1)

        select_idx = random_subset[torch.Tensor(-np.array(_uncertainties)).sort(descending=True)[1][:self.n_query]]

        return select_idx    