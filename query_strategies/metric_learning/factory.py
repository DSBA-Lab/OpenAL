import numpy as np
import os
import torch
from copy import deepcopy
from tqdm.auto import tqdm

from query_strategies.utils import torch_seed


def create_metric_learning(
    method_name, vis_encoder, criterion, epochs: int, opt_name: str, lr: float, 
    seed: int = 223, opt_params: dict = {}, **kwargs):
    
    metric_learning = __import__('query_strategies.metric_learning', fromlist='metric_learning').__dict__[method_name](
        vis_encoder     = vis_encoder, 
        criterion       = criterion, 
        epochs          = epochs, \
        opt_name        = opt_name, 
        lr              = lr, 
        seed            = seed, 
        opt_params      = opt_params,
        **kwargs
    )
    
    return metric_learning


class MetricLearning:
    def __init__(
        self, vis_encoder, criterion, epochs: int, opt_name: str, lr: float, 
        savepath: str = None, seed: int = 223, opt_params: dict = {}, **kwargs):
        
        self.vis_encoder = vis_encoder
        self.criterion = criterion
        self.optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name]
        self.lr = lr
        self.opt_params = opt_params
        
        self.epochs = epochs
        self.seed = seed
        self.savepath = savepath
        
    def init_model(self, device: str):
        vis_encoder = deepcopy(self.vis_encoder)
        
        if self.savepath != None and os.path.isfile(self.savepath):    
            vis_encoder.load_state_dict(torch.load(self.savepath))
            
        return vis_encoder.to(device)
        
        
    def fit(self, vis_encoder, dataset, sample_idx: np.ndarray, device: str, **kwargs):
        torch_seed(self.seed)
        
        # split dataset
        self.create_trainset(dataset=dataset, sample_idx=sample_idx, **kwargs)

        # optimizer
        optimizer = self.optimizer(vis_encoder.parameters(), lr=self.lr, **self.opt_params)
                
        for _ in tqdm(range(self.epochs), total=self.epochs, desc='Metric Learning'):
            self.train(vis_encoder=vis_encoder, optimizer=optimizer, device=device)
            
        self.vis_encoder.eval()
        
        if self.savepath:
            torch.save(self.vis_encoder.state_dict(), self.savepath)
        
    def create_trainset(self):
        raise NotImplementedError
        
    
    def train(self):
        raise NotImplementedError
            
    
    def test(self) -> float:
        raise NotImplementedError