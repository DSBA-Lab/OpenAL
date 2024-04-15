import torch
from tqdm.auto import tqdm

from query_strategies.utils import torch_seed


def create_metric_learning(
    method_name, 
    savepath: str, 
    seed: int = 223, 
    accelerator = None,
    **kwargs
):  
    metric_learning = __import__('metric_learning').__dict__[method_name](
        savepath    = savepath,
        seed        = seed, 
        accelerator = accelerator,
        **kwargs
    )
    
    return metric_learning


class MetricLearning:
    def __init__(
        self, 
        savepath: str, 
        seed: int = 223, 
        accelerator = None,
        **kwargs
    ):
        self.accelerator = accelerator
        
        # save
        self.seed = seed
        self.savepath = savepath
        
    def fit(self, epochs: int, vis_encoder, dataloader, optimizer, scheduler, device: str, **kwargs):
        torch_seed(self.seed)
        
        if self.accelerator != None:
            vis_encoder, dataloader, optimizer, scheduler = self.accelerator.prepare(
                vis_encoder, dataloader, optimizer, scheduler
            )
                
        desc = '[{name}] lr: {lr:.3e}'
        p_bar = tqdm(range(epochs), total=epochs)
        
        for epoch in p_bar:
            p_bar.set_description(
                desc=desc.format(
                    name = self.__class__.__name__, 
                    lr   = optimizer.param_groups[0]['lr']
                )
            )
            self.train(
                epoch       = epoch,
                vis_encoder = vis_encoder,
                dataloader  = dataloader,
                optimizer   = optimizer,
                scheduler   = scheduler,
                device      = device
            )
            scheduler.step()
            
        vis_encoder.eval()
        
        torch.save(vis_encoder.state_dict(), self.savepath)
        
    def create_trainset(self):
        raise NotImplementedError
        
    
    def train(self):
        raise NotImplementedError
        