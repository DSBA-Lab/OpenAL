import os
import wandb
import logging
import sys

from torch.utils.data import DataLoader

from accelerate import Accelerator
from omegaconf import OmegaConf

from arguments import parser
from datasets import create_dataset
from log import setup_default_logging
from metric_learning import create_metric_learning, MetricModel
from query_strategies import torch_seed
from query_strategies.scheds import create_scheduler
from query_strategies.optims import create_optimizer

_logger = logging.getLogger('train')

def run(cfg):
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )

    setup_default_logging()
    torch_seed(cfg.DEFAULT.seed)

    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # load dataset
    trainset, _, _ = create_dataset(
        datadir  = cfg.DATASET.datadir, 
        dataname = cfg.DATASET.name,
        img_size = cfg.DATASET.img_size,
        mean     = cfg.DATASET.mean,
        std      = cfg.DATASET.std,
        aug_info = cfg.DATASET.aug_info,
        **cfg.DATASET.get('params', {})
    )
    trainloader = DataLoader(
        trainset, 
        batch_size  = cfg.DATASET.batch_size, 
        num_workers = cfg.DATASET.num_workers,
    )
    
    # make save directory
    savedir = os.path.join(cfg.DEFAULT.savedir, cfg.DATASET.name)
    savepath = os.path.join(savedir, f'{cfg.MODEL.name}_{cfg.SSL.method}.pt')
    
    assert not os.path.isfile(savepath), f'{savepath} already exists'
    os.makedirs(savedir, exist_ok=True)
    
    # save configs
    OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
    
    # initialize wandb
    if cfg.TRAIN.wandb.use:
        wandb.init(name=cfg.DEFAULT.exp_name, project=cfg.TRAIN.wandb.project_name, entity=cfg.TRAIN.wandb.entity, config=OmegaConf.to_container(cfg))
    
    # metric learning
    vis_encoder = MetricModel(
        modelname   = cfg.MODEL.name,
        pretrained  = cfg.MODEL.pretrained,
        **cfg.MODEL.get('params', {})
    )
    
    # optimizer
    optimizer = create_optimizer(opt_name=cfg.OPTIMIZER.name, model=vis_encoder, lr=cfg.OPTIMIZER.lr, opt_params=cfg.OPTIMIZER.params)

    scheduler = create_scheduler(
        sched_name    = cfg.SCHEDULER.name, 
        optimizer     = optimizer,
        epochs        = cfg.TRAIN.epochs,
        params        = cfg.SCHEDULER.params,
        warmup_params = cfg.SCHEDULER.get('warmup_params', {})
    )
    
    SSLTrainer = create_metric_learning(
        method_name = cfg.SSL.method,
        savepath    = savepath, 
        accelerator = accelerator,
        seed        = cfg.DEFAULT.seed, 
        dataname    = cfg.DATASET.name,
        img_size    = cfg.DATASET.img_size,
        ssl_params  = cfg.SSL.get('params', {})
    )
    
    SSLTrainer.fit(
        epochs      = cfg.TRAIN.epochs,
        vis_encoder = vis_encoder,
        dataloader  = trainloader,
        optimizer   = optimizer,
        scheduler   = scheduler,
        device      = accelerator.device
    )
    

if __name__=='__main__':
    # CSI training
    sys.setrecursionlimit(10000)

    # config
    cfg = parser()
    
    # run
    run(cfg)