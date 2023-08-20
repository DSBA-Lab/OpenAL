import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from .strategy import Strategy

class KCenterGreedy(Strategy):
    def __init__(self, model, n_query: int, labeled_idx: np.ndarray, 
                 dataset: Dataset, batch_size: int, num_workers: int, n_subset: int = 0):
        
        super(KCenterGreedy, self).__init__(
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
        
        # predict probability and embedding on unlabeled dataset
        ulb_embed = self.extract_outputs(
            model        = model, 
            sample_idx   = unlabeled_idx, 
            return_probs = False,
            return_embed = True
        )['embed']
        
        ulb_embed = ulb_embed.flatten(start_dim=1)
        
        # predict probability and embedding on labeled dataset
        lb_embed = self.extract_outputs(
            model        = model, 
            sample_idx   = np.where(self.labeled_idx==True)[0], 
            return_probs = False,
            return_embed = True
        )['embed']
        
        lb_embed = lb_embed.flatten(start_dim=1)
        
        # distance matrix between unlabeled and labeled embeddings
        mat = self._distance(embed1=ulb_embed, embed2=lb_embed)

        # K-center greedy
        selected_idx = []
        labeled_idx = deepcopy(self.labeled_idx)
        
        pbar = tqdm(range(self.n_query), desc=f"K-center Greedy: {mat.shape}")
        
        for _ in pbar:
            # nearest distance of unlabeled data wrt labeled data
            mat_min, _ = mat.min(dim=1) # return (values, indexes)
            
            # index with largest distance among unlabeled data
            q_idx_ = mat_min.argmax().item()
            
            # find index from unlabeled pool
            q_idx = np.arange(len(self.labeled_idx))[labeled_idx==False][q_idx_]
            
            # change selected index into labeled pool
            labeled_idx[q_idx] = True
            selected_idx.append(q_idx)
            
            # delete selected index from distance matrix
            mat = torch.cat([mat[:q_idx_] , mat[q_idx_ + 1:]], dim=0)
            
            # append selected index into labeled column of distance matrix 
            ulb_embed, new_dist = self.update_ulb_embed(select_idx=q_idx_, ulb_embed=ulb_embed)            
            mat = torch.cat([mat, new_dist], dim=1)
            
            pbar.set_description(f"K-center Greedy: {mat.shape}")
            
        return np.array(selected_idx)
    
    def update_ulb_embed(self, select_idx: int, ulb_embed: torch.Tensor):
        selected_embed = ulb_embed[select_idx].unsqueeze(0)
        ulb_embed = torch.cat([ulb_embed[:select_idx] , ulb_embed[select_idx + 1:]], dim=0)
        new_dist = self._distance(embed1 = ulb_embed, embed2 = selected_embed)
        
        return ulb_embed, new_dist
    
    def _distance(self, embed1: torch.Tensor, embed2: torch.Tensor):
        # matrix multiplcation
        mat = torch.matmul(embed1, embed2.transpose(1,0))
        
        # \sqrt{ulb_embed^2 + lb_embed^2 -2 * (ulb_embed * lb_embed)}
        mat = ((mat * -2) + (embed1**2).sum(dim=1).unsqueeze(1) + (embed2**2).sum(dim=1).unsqueeze(0)).sqrt()
        
        return mat