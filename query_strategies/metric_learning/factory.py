import numpy as np
import os
import torch
from copy import deepcopy
from tqdm.auto import tqdm

from query_strategies.scheds import create_scheduler
from query_strategies.optims import create_optimizer
from query_strategies.utils import torch_seed


def create_metric_learning(
    method_name, 
    vis_encoder, 
    epochs: int, 
    opt_name: str, 
    lr: float, 
    sched_name: str, 
    sched_params: dict, 
    warmup_params: dict = {},
    seed: int = 223, 
    opt_params: dict = {}, 
    accelerator = None,
    **kwargs
):
    
    metric_learning = __import__('query_strategies.metric_learning', fromlist='metric_learning').__dict__[method_name](
        vis_encoder     = vis_encoder, 
        epochs          = epochs, 
        opt_name        = opt_name, 
        lr              = lr, 
        seed            = seed, 
        opt_params      = opt_params,
        sched_name      = sched_name,
        sched_params    = sched_params,
        warmup_params   = warmup_params,
        accelerator     = accelerator,
        **kwargs
    )
    
    return metric_learning


class MetricLearning:
    def __init__(
        self, 
        vis_encoder, 
        epochs: int,
        opt_name: str, 
        lr: float, 
        sched_name: str,
        sched_params: dict,
        warmup_params: dict = {},
        opt_params: dict = {}, 
        aug_info: dict = None,
        savepath: str = None, 
        seed: int = 223, 
        accelerator = None,
        **kwargs
    ):
        self.accelerator = accelerator
        
        self.vis_encoder = vis_encoder
        self.epochs = epochs
        
        # optimizer
        self.opt_name = opt_name
        self.lr = lr
        self.opt_params = opt_params
        
        # scheduler
        self.sched_name = sched_name
        self.sched_params = sched_params
        self.warmup_params = warmup_params
        
        # save
        self.seed = seed
        self.savepath = savepath
        
    def init_model(self, device: str):
        vis_encoder = deepcopy(self.vis_encoder)
        
        if self.savepath != None and os.path.isfile(self.savepath):    
            vis_encoder.load_state_dict(torch.load(self.savepath))
            print('load metric model from {}'.format(self.savepath))
            
        if self.accelerator != None:
            vis_encoder = self.accelerator.prepare(vis_encoder)
        else:
            vis_encoder.to(device)
        vis_encoder.eval()
            
        return vis_encoder
        
        
    def fit(self, vis_encoder, dataset, sample_idx: np.ndarray, device: str, **kwargs):
        torch_seed(self.seed)
        
        # split dataset
        self.create_trainset(dataset=dataset, sample_idx=sample_idx, **kwargs)

        # optimizer
        optimizer = create_optimizer(opt_name=self.opt_name, model=vis_encoder, lr=self.lr, opt_params=self.opt_params)

        scheduler = create_scheduler(
            sched_name    = self.sched_name, 
            optimizer     = optimizer,
            epochs        = self.epochs,
            params        = self.sched_params,
            warmup_params = self.warmup_params
        )
        
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)
                
        desc = '[{name}] lr: {lr:.3e}'
        p_bar = tqdm(range(self.epochs), total=self.epochs)
        
        for epoch in p_bar:
            p_bar.set_description(
                desc=desc.format(
                    name = self.__class__.__name__, 
                    lr   = optimizer.param_groups[0]['lr']
                )
            )
            self.train(epoch=epoch, vis_encoder=vis_encoder, optimizer=optimizer, scheduler=scheduler, device=device)
            scheduler.step()
            
        vis_encoder.eval()
        
        if self.savepath:
            torch.save(vis_encoder.state_dict(), self.savepath)
        
    def create_trainset(self):
        raise NotImplementedError
        
    
    def train(self):
        raise NotImplementedError
        