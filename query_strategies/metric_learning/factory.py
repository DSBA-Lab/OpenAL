import numpy as np

from copy import deepcopy
from tqdm.auto import tqdm

from query_strategies.utils import torch_seed


def create_metric_learning(
    method_name, vis_encoder, criterion, epochs: int, test_ratio: float, opt_name: str, lr: float, savedir: str, 
    train_transform, test_transform = None, seed: int = 223, opt_params: dict = {}, **kwargs):
    
    metric_learning = __import__('query_strategies.metric_learning', fromlist='metric_learning').__dict__[method_name](
        vis_encoder     = vis_encoder, 
        criterion       = criterion, 
        epochs          = epochs, 
        train_transform = train_transform, 
        test_transform  = test_transform,
        test_ratio      = test_ratio, 
        opt_name        = opt_name, 
        lr              = lr, 
        savedir         = savedir, 
        seed            = seed, 
        opt_params      = opt_params,
        **kwargs
    )
    
    return metric_learning


class MetricLearning:
    def __init__(
        self, vis_encoder, criterion, epochs: int, test_ratio: float, opt_name: str, lr: float, savedir: str, 
        train_transform, test_transform = None, seed: int = 223, opt_params: dict = {}, **kwargs):
        
        self.vis_encoder = vis_encoder
        self.criterion = criterion
        self.optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name]
        self.lr = lr
        self.opt_params = opt_params
        
        self.test_ratio = test_ratio
        self.train_transform = train_transform
        self.test_transform = test_transform
        
        self.epochs = epochs
        self.savedir = savedir
        self.seed = seed
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        

    def init_model(self, device: str):
        return deepcopy(self.vis_encoder).to(device)
        
    def fit(self, vis_encoder, dataset, sample_idx: np.ndarray, device: str, **kwargs):
        torch_seed(self.seed)
        
        # split dataset
        self.create_trainset(dataset=dataset, sample_idx=sample_idx, **kwargs)

        # optimizer
        optimizer = self.optimizer(vis_encoder.parameters(), lr=self.lr, **self.opt_params)
                
        for _ in tqdm(range(self.epochs), total=self.epochs):
            self.train(vis_encoder=vis_encoder, optimizer=optimizer, device=device)
        
    def create_trainset(self):
        raise NotImplementedError
        
    
    def train(self):
        raise NotImplementedError
            
    
    def test(self) -> float:
        raise NotImplementedError