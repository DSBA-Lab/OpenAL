import torch
from .warmup import GradualWarmupScheduler

def create_scheduler(sched_name: str, optimizer, epochs: int, params: dict, warmup_params: dict = {}):
    if sched_name == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=params['t_mult'], eta_min=params['eta_min'])
    elif sched_name == 'multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['milestones'])
    elif sched_name == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
        
    # add warmup
    if warmup_params.get('use'):
        scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_params['multiplier'], total_epoch=warmup_params['warmup'], after_scheduler=scheduler)
        
    return scheduler