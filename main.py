import os
import wandb
import logging
from arguments import parser

from train import al_run, full_run
from datasets import create_dataset
from log import setup_default_logging
from utils import torch_seed

from accelerate import Accelerator
from omegaconf import OmegaConf

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
    trainset, validset, testset = create_dataset(
        datadir  = cfg.DATASET.datadir, 
        dataname = cfg.DATASET.name,
        img_size = cfg.DATASET.img_size,
        mean     = cfg.DATASET.mean,
        std      = cfg.DATASET.std,
        aug_info = cfg.DATASET.aug_info,
        seed     = cfg.DATASET.seed
    )
    
    if 'AL' in cfg.keys():
        # make save directory
        al_name = f"total_{cfg.AL.n_end}-init_{cfg.AL.n_start}-query_{cfg.AL.n_query}"
        savedir = os.path.join(
            cfg.DEFAULT.savedir, cfg.DATASET.name, cfg.MODEL.name, 
            cfg.AL.strategy, cfg.DEFAULT.exp_name, al_name, f'seed{cfg.DEFAULT.seed}'
        )
        
        assert not os.path.isdir(savedir), f'{savedir} already exists'
        os.makedirs(savedir)
        
        # save config
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        
        # run active learning
        al_run(
            cfg         = cfg,
            trainset    = trainset,
            validset    = validset,
            testset     = testset,
            savedir     = savedir,
            accelerator = accelerator,
        )
    else:
        # make save directory
        savedir = os.path.join(cfg.DEFAULT.savedir, cfg.DATASET.name, cfg.MODEL.name, 'Full', cfg.DEFAULT.exp_name)
        
        assert not os.path.isdir(savedir), f'{savedir} already exists'
        os.makedirs(savedir)
        
        # save configs
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        
        # initialize wandb
        if cfg.TRAIN.wandb.use:
            wandb.init(name=cfg.DEFAULT.exp_name, project=cfg.TRAIN.wandb.project_name, entity=cfg.TRAIN.wandb.entity, config=OmegaConf.to_container(cfg))
        
        # run full supervised learning
        full_run(
            cfg         = cfg,
            trainset    = trainset,
            validset    = validset,
            testset     = testset,
            savedir     = savedir,
            accelerator = accelerator,
        )
    

if __name__=='__main__':

    # config
    cfg = parser()
    
    # run
    run(cfg)