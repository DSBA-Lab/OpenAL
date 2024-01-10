import numpy as np
import torch
import os
from tqdm.auto import tqdm

from .strategy import Strategy

class PromptEnsemble(Strategy):
    def __init__(self, agg_type: str, ensemble_type: str, weights: float = 0.5, **init_args):
        super(PromptEnsemble, self).__init__(**init_args)
        self.agg_type = agg_type
        self.ensemble_type = ensemble_type
        self.weights = weights
        
    def query(self, model, r: int, seed: int, savedir: str) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
       
        if self.ensemble_type == 'ind': 
            total_probs = []
        
            for r_i in tqdm(range(r), desc='Inference round weights'):
                prompt_weights = torch.load(os.path.join(savedir, f"prompt_seed{seed}-round{r_i}.pt"))
                model.prompt.load_state_dict(prompt_weights)
            
                # predict probability on unlabeled dataset
                probs = self.extract_outputs(
                    model      = model, 
                    sample_idx = unlabeled_idx, 
                )['probs']
                
                total_probs.append(probs)
            
            total_probs = torch.stack(total_probs)
            
            if self.agg_type == 'mean':
                probs = total_probs.mean(dim=0)
                
        elif self.ensemble_type == 'avg':
            # weight average
            prompt_weights = self.weights_average(r=r, seed=seed, savedir=savedir, weights=self.weights)
            model.prompt.load_state_dict(prompt_weights)
        
            # predict probability on unlabeled dataset
            probs = self.extract_outputs(
                model      = model, 
                sample_idx = unlabeled_idx, 
            )['probs']
        
        # select maximum entropy
        entropy = (-(probs*torch.log(probs))).sum(dim=1)
        select_idx = unlabeled_idx[entropy.sort(descending=True)[1][:self.n_query]]
        
        return select_idx
    
    @staticmethod
    def weights_average(r: int, seed: int, savedir: str, weights: float):
        # weight average
        prompt_weights = torch.load(os.path.join(savedir, f"prompt_seed{seed}-round0.pt"))
        for r_i in tqdm(range(1, r), desc='Average round weights'):
            new_w = torch.load(os.path.join(savedir, f"prompt_seed{seed}-round{r_i}.pt"))
        
            for k in prompt_weights.keys():
                prompt_weights[k].data = weights * prompt_weights[k].data + (1 - weights) * new_w[k].data
            
        return prompt_weights
        