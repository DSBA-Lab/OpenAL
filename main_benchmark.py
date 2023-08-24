import numpy as np
import os
import random
import wandb
import torch
import logging
from arguments import parser

from train import al_run, full_run
from log import setup_default_logging

from accelerate import Accelerator
from omegaconf import OmegaConf

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


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
    trainset, testset = __import__('datasets').__dict__[f"load_{cfg.DATASET.dataname.lower()}"](
        datadir  = cfg.DATASET.datadir, 
        img_size = cfg.DATASET.img_size,
        mean     = cfg.DATASET.mean, 
        std      = cfg.DATASET.std,
        aug_info = cfg.DATASET.aug_info,
        **cfg.DATASET.get('params', {})
    )
    
    if 'AL' in cfg.keys():
        # make save directory
        al_name = f"total_{cfg.AL.n_end}-init_{cfg.AL.n_start}-query_{cfg.AL.n_query}"
        savedir = os.path.join(
            cfg.DEFAULT.savedir, cfg.DATASET.dataname, cfg.MODEL.modelname, 
            cfg.AL.strategy, cfg.DEFAULT.exp_name, al_name, f'seed{cfg.DEFAULT.seed}'
        )
        os.makedirs(savedir, exist_ok=True)
        
        # save config
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        
        # run active learning
        al_run(
            exp_name           = cfg.DEFAULT.exp_name, 
            modelname          = cfg.MODEL.modelname,
            pretrained         = cfg.MODEL.pretrained,
            strategy           = cfg.AL.strategy,
            n_start            = cfg.AL.n_start,
            n_end              = cfg.AL.n_end,
            n_query            = cfg.AL.n_query,
            n_subset           = cfg.AL.n_subset,
            init_method        = cfg.AL.init.method,
            init_method_params = cfg.AL.init.get('params', {}),
            trainset           = trainset,
            validset           = testset,
            testset            = testset,
            img_size           = cfg.DATASET.img_size,
            num_classes        = cfg.DATASET.num_classes,
            batch_size         = cfg.DATASET.batch_size,
            test_batch_size    = cfg.DATASET.test_batch_size,
            num_workers        = cfg.DATASET.num_workers,
            opt_name           = cfg.OPTIMIZER.opt_name,
            lr                 = cfg.OPTIMIZER.lr,
            opt_params         = cfg.OPTIMIZER.get('params',{}),
            sched_name         = cfg.SCHEDULER.sched_name,
            sched_params       = cfg.SCHEDULER.params,
            epochs             = cfg.TRAIN.epochs,
            log_interval       = cfg.TRAIN.log_interval,
            use_wandb          = cfg.TRAIN.wandb.use,
            savedir            = savedir,
            seed               = cfg.DEFAULT.seed,
            accelerator        = accelerator,
            cfg                = cfg
        )
    else:
        # make save directory
        savedir = os.path.join(cfg.DEFAULT.savedir, cfg.DATASET.dataname, cfg.MODEL.modelname, 'Full', cfg.DEFAULT.exp_name)
        os.makedirs(savedir, exist_ok=True)
        
        # save configs
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        
        # initialize wandb
        if cfg.TRAIN.wandb.use:
            wandb.init(name=cfg.DEFAULT.exp_name, project=cfg.TRAIN.wandb.project_name, entity=cfg.TRAIN.wandb.entity, config=OmegaConf.to_container(cfg))
        
        # run full supervised learning
        full_run(
            modelname       = cfg.MODEL.modelname,
            pretrained      = cfg.MODEL.pretrained,
            trainset        = trainset,
            validset        = testset,
            testset         = testset,
            img_size        = cfg.DATASET.img_size,
            num_classes     = cfg.DATASET.num_classes,
            batch_size      = cfg.DATASET.batch_size,
            test_batch_size = cfg.DATASET.test_batch_size,
            num_workers     = cfg.DATASET.num_workers,
            opt_name        = cfg.OPTIMIZER.opt_name,
            lr              = cfg.OPTIMIZER.lr,
            opt_params      = cfg.OPTIMIZER.params,
            sched_name      = cfg.SCHEDULER.sched_name,
            sched_params    = cfg.SCHEDULER.params,
            epochs          = cfg.TRAIN.epochs,
            log_interval    = cfg.TRAIN.log_interval,
            use_wandb       = cfg.TRAIN.wandb.use,
            savedir         = savedir,
            seed            = cfg.DEFAULT.seed,
            accelerator     = accelerator
        )
    
    
if __name__=='__main__':

    # config
    cfg = parser()
    
    # run
    run(cfg)