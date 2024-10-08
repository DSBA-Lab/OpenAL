import torch
from .warmup import GradualWarmupScheduler

def create_scheduler(sched_name: str, optimizer, epochs: int, params: dict, warmup_params: dict = {}):
    
    if isinstance(optimizer, dict):
        scheduler = {}
        for k, opt in optimizer.items():
            scheduler[k] = _create_scheduler(
                sched_name    = sched_name,
                optimizer     = opt,
                epochs        = epochs,
                params        = params,
                warmup_params = warmup_params
            )
    else:
        scheduler = _create_scheduler(
            sched_name    = sched_name,
            optimizer     = optimizer,
            epochs        = epochs,
            params        = params,
            warmup_params = warmup_params
        )
        
    return scheduler

def _create_scheduler(sched_name: str, optimizer, epochs: int, params: dict, warmup_params: dict = {}):
    if sched_name == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=params['t_mult'], eta_min=params['eta_min'])
    elif sched_name == 'multi_step':
        default_milstones = [int(epochs*0.5), int(epochs*0.75)]
        milestones = params.get('milestones', default_milstones)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    elif sched_name == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
        
    # add warmup
    if warmup_params.get('use'):
        scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_params['multiplier'], total_epoch=warmup_params['warmup'], after_scheduler=scheduler)
        
    return scheduler