import numpy as np
import torch
from copy import deepcopy
from tqdm.auto import tqdm
from .strategy import Strategy

class KCenterGreedy(Strategy):
    def __init__(self, **init_args):
        
        super(KCenterGreedy, self).__init__(**init_args)
        
    def query(self, model, **kwargs) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = kwargs.get('unlabeled_idx', self.get_unlabeled_idx())
        
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
            sample_idx   = np.where(self.is_labeled==True)[0], 
            return_probs = False,
            return_embed = True
        )['embed']
        
        lb_embed = lb_embed.flatten(start_dim=1)
        
        # distance matrix between unlabeled and labeled embeddings
        mat = self._distance(embed1=ulb_embed, embed2=lb_embed)

        # greedy selection
        selected_idx = self.greedy_selection(mat=mat, ulb_embed=ulb_embed, unlabeled_idx=unlabeled_idx)
        
        return selected_idx


    def greedy_selection(self, mat: torch.Tensor, ulb_embed: torch.Tensor, unlabeled_idx: np.ndarray):
        # K-center greedy
        selected_idx = []
        is_labeled = deepcopy(self.is_labeled)
        
        pbar = tqdm(range(self.n_query), desc=f"K-center Greedy: {mat.shape}")
        
        for _ in pbar:
            # nearest distance of unlabeled data wrt labeled data
            mat_min, _ = mat.min(dim=1) # return (values, indexes)
            
            # index with largest distance among unlabeled data
            q_idx_ = mat_min.argmax().item()
            
            # find index from unlabeled pool
            q_idx = np.arange(len(self.is_labeled))[unlabeled_idx][q_idx_]
            
            # change selected index into labeled pool
            is_labeled[q_idx] = True
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
    
    
    
class KCenterGreedyCB(KCenterGreedy):
    """
    Class-Balanced Active Learning for Image Classification. WACV 2022
    """
    def __init__(self, lamb: int = 5, **init_params):
        
        # round log
        self.r = 1
        
        # lambda
        self.lamb = lamb
        
        super(KCenterGreedyCB, self).__init__(**init_params)
        
    def query(self, model, **kwargs) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = kwargs.get('unlabeled_idx', self.get_unlabeled_idx())
        
        # predict probability and embedding on unlabeled dataset
        ulb_outputs = self.extract_outputs(
            model        = model, 
            sample_idx   = unlabeled_idx, 
            return_probs = True,
            return_embed = True
        )
        
        ulb_embed, ulb_probs = ulb_outputs['embed'], ulb_outputs['probs']
        ulb_embed = ulb_embed.flatten(start_dim=1)
        
        # predict probability and embedding on labeled dataset
        lb_outputs = self.extract_outputs(
            model         = model, 
            sample_idx    = np.where(self.is_labeled==True)[0], 
            return_probs  = False,
            return_embed  = True,
            return_labels = True
        )
        
        lb_embed, lb_labels = lb_outputs['embed'], lb_outputs['labels']
        lb_embed = lb_embed.flatten(start_dim=1)
        
        # distance matrix between unlabeled and labeled embeddings
        mat = self._distance(embed1=ulb_embed, embed2=lb_embed)

        # greedy selection
        selected_idx = self.class_balanced_greedy_selection(
            mat           = mat, 
            ulb_embed     = ulb_embed, 
            ulb_probs     = ulb_probs, 
            lb_labels     = lb_labels,
            r             = self.r,
            lamb          = self.lamb,
            unlabeled_idx = unlabeled_idx
        )
        
        return selected_idx
    
    
    def class_balanced_greedy_selection(
        self, mat: torch.Tensor, unlabeled_idx: np.ndarray, ulb_embed: torch.Tensor, ulb_probs: torch.Tensor, lb_labels: torch.Tensor, r: int, lamb: int):
        
        # Class-balance K-center greedy
        
        selected_idx = []
        is_labeled = deepcopy(self.is_labeled)
        
        # the number of samples per class
        _, counts = torch.unique(lb_labels, return_counts=True)
        num_classes = len(counts)
        
        # samples required per class
        num_required = int((sum(counts)+(r*self.n_query))/len(counts)) - counts
        num_required = torch.clamp(num_required, min=0).unsqueeze(1)
        
        # selected unlabeled samples
        z = torch.zeros(ulb_embed.size(0), dtype=bool)
        
        # Q is probs of unlabeled samples
        q = deepcopy(ulb_probs)
        
        pbar = tqdm(range(self.n_query), desc=f"K-center Greedy (CB): {mat.shape}")
        
        for _ in pbar:
            # the number of unlabeled samples
            ulb_size = ulb_embed.size(0)
            
            # nearest distance of unlabeled data wrt labeled data
            mat_min, _ = mat.min(dim=1) # return (values, indexes), mat_min: (ulb_size, )
            
            # repeat num_required as unlabeled size
            num_required_repeat = torch.tile(input=num_required, dims=(1, ulb_size)) # num_required_repeat: (num classes, ulb_size)
            
            # ulb_probs_z is P*z that is marginal probs of selected samples per class
            ulb_probs_z = torch.tile(input=torch.matmul(ulb_probs.transpose(1,0), z.to(torch.float)), dims=(ulb_size, 1)) # ulb_probs_z: (ulb_size, num_classes)
            
            # ||num_required_repeat - q - ulb_probs_z||_1 is L1 loss between num_required_repeat and Q + ulb_probs_z.
            # Q + ulb_probs_z means selected samples' probs added into unlabeled samples' probs. 
            # the lower the L1 loss, the more samples are needed for class balance.
            l1_loss = torch.linalg.norm(num_required_repeat - (q.transpose(1,0) + ulb_probs_z.transpose(1,0)), dim=0, ord=1) # l1_loss: (ulb_size, )
            
            # \argmin_z{-mat_min + \frec{\lambda}{num_classes} * l1_loss}. index with largest distance among unlabeled data w.r.t. num_required per class and ulb_probs
            q_idx_ = (-mat_min + (lamb / num_classes) * l1_loss).argmin().item()
            
            # find index from unlabeled pool
            q_idx = np.arange(len(self.is_labeled))[unlabeled_idx][q_idx_]
        
            # change selected index into unlabeled pool
            z_idx = np.arange(z.size(0))[z==False][q_idx_]
            z[z_idx] = True
            
            # change selected index into labeled pool
            is_labeled[q_idx] = True
            selected_idx.append(q_idx)
            
            # delete selected index from probs of unlabeled samples
            q = deepcopy(ulb_probs[z==False])
            
            # delete selected index from distance matrix
            mat = torch.cat([mat[:q_idx_] , mat[q_idx_ + 1:]], dim=0)
            
            # append selected index into labeled column of distance matrix 
            ulb_embed, new_dist = self.update_ulb_embed(select_idx=q_idx_, ulb_embed=ulb_embed)
            mat = torch.cat([mat, new_dist], dim=1)
            
            pbar.set_description(f"K-center Greedy (CB): {mat.shape}")
            
        return np.array(selected_idx)