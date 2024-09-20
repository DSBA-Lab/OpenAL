import os
import wandb
from omegaconf import OmegaConf

from arguments import parser
from train import al_run, openset_al_run, full_run
from datasets import create_dataset
from log import setup_default_logging
from query_strategies import torch_seed

def make_directory(savedir: str, is_resume: bool = False):
    assert not os.path.isdir(savedir) or is_resume, f'{savedir} already exists'
    if not os.path.isdir(savedir) or not is_resume:
        os.makedirs(savedir)

def run(cfg):
    
    setup_default_logging()
    torch_seed(cfg.DEFAULT.seed)

    # load dataset
    trainset, validset, testset = create_dataset(
        datadir  = cfg.DATASET.datadir, 
        dataname = cfg.DATASET.name,
        img_size = cfg.DATASET.img_size,
        mean     = cfg.DATASET.mean,
        std      = cfg.DATASET.std,
        aug_info = cfg.DATASET.aug_info,
        **cfg.DATASET.get('params', {})
    )
    
    if 'AL' in cfg:
        # make save directory
        al_name = f"total_{cfg.AL.n_end}-init_{cfg.AL.n_start}-query_{cfg.AL.n_query}"
        savedir = os.path.join(
            cfg.DEFAULT.savedir, cfg.DATASET.name, cfg.MODEL.name, 
            cfg.AL.strategy, cfg.DEFAULT.exp_name, al_name, f'seed{cfg.DEFAULT.seed}'
        )
        
        make_directory(
            savedir   = savedir,
            is_resume = cfg.TRAIN.get('resume', False).get('use', False)
        )
        
        # save config
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        
        # run active learning
        if 'ood_ratio' in cfg.AL:
            openset_al_run(
                cfg         = cfg,
                trainset    = trainset,
                validset    = validset,
                testset     = testset,
                savedir     = savedir,
            )        
        else:
            al_run(
                cfg         = cfg,
                trainset    = trainset,
                validset    = validset,
                testset     = testset,
                savedir     = savedir,
            )
    else:
        # make save directory
        savedir = os.path.join(cfg.DEFAULT.savedir, cfg.DATASET.name, cfg.MODEL.name, 'Full', cfg.DEFAULT.exp_name, f'seed{cfg.DEFAULT.seed}')
        
        make_directory(
            savedir   = savedir,
            is_resume = cfg.TRAIN.get('resume', False).get('use', False)
        )
        
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
            savedir     = savedir
        )
    

if __name__=='__main__':
    # config
    cfg = parser()
    
    # run
    run(cfg)