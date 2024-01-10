import numpy as np
import torch
from tqdm.auto import tqdm
from collections import defaultdict
from .strategy import Strategy

class PromptUncertainty(Strategy):
    def __init__(self, loss_weight: float, **init_args):
        super(PromptUncertainty, self).__init__(**init_args)
        
        self.loss_weight = loss_weight
        
    def query(self, model) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
        
        # predict probability on unlabeled dataset
        outputs = self.extract_outputs(
            model      = model, 
            sample_idx = unlabeled_idx, 
        )
        
        # select maximum entropy
        uncertainty = (outputs['probs'] * outputs['prompt_probs']).max(dim=1)[0]
        select_idx = unlabeled_idx[uncertainty.sort(descending=True)[1][:self.n_query]]
        
        return select_idx
    

    def loss_fn(self, outputs, targets):
        ce_loss = self.criterion(outputs['logits'], targets)
        prompt_ce_loss = self.criterion(outputs['prompt_logits'], targets)
    
        return ce_loss.mean() + (self.loss_weight * prompt_ce_loss.mean())
    
    def get_outputs(
        self, model, dataloader, device: str, **kwargs) -> dict:
    
        # predict
        results = defaultdict(list)
    
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Get outputs [Probs, Prompt Probs]', leave=False):
                if len(batch) == 2:
                    # for labeled dataset that contains labels
                    inputs, targets = batch
                else:
                    # for unlabeled dataset that does not contain labels
                    inputs = batch
                    
                outputs = model(inputs.to(device))
                if self.tta != None:
                    prompt_logits = outputs['prompt_logits']
                    logits = self.tta_outputs(model=model, inputs=inputs, device=device)
                else:
                    logits = outputs['logits']
                    prompt_logits = outputs['prompt_logits']
                    
                probs = torch.nn.functional.softmax(logits, dim=1)
                prompt_probs = torch.nn.functional.softmax(prompt_logits, dim=1)
                results['probs'].append(probs.cpu())
                results['prompt_probs'].append(prompt_probs.cpu())
        
        results['probs'] = torch.vstack(results['probs'])
        results['prompt_probs'] = torch.vstack(results['prompt_probs'])
    
        return results