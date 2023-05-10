import numpy as np
import os
import random
import wandb
import torch
import argparse
import yaml
import logging
import json 

from train import al_run, full_run
from datasets import stats
from log import setup_default_logging

from accelerate import Accelerator

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
        gradient_accumulation_steps = cfg['TRAIN']['grad_accum_steps'],
        mixed_precision             = cfg['TRAIN']['mixed_precision']
    )

    setup_default_logging()
    torch_seed(cfg['DEFAULT']['seed'])

    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # load dataset
    trainset, testset = __import__('datasets').__dict__[f"load_{cfg['DATASET']['dataname'].lower()}"](
        datadir            = cfg['DATASET']['datadir'], 
        img_size           = cfg['DATASET']['img_size'],
        mean               = cfg['DATASET']['mean'], 
        std                = cfg['DATASET']['std']
    )
    
    if 'AL' in cfg.keys():
        # make save directory
        al_name = f"total_{cfg['AL']['n_end']}-init_{cfg['AL']['n_start']}-query_{cfg['AL']['n_query']}"
        savedir = os.path.join(cfg['DEFAULT']['savedir'], cfg['DATASET']['dataname'], cfg['MODEL']['modelname'], cfg['DEFAULT']['exp_name'], al_name)
        os.makedirs(savedir, exist_ok=True)
        
        # run active learning
        al_run(
            exp_name        = cfg['DEFAULT']['exp_name'], 
            modelname       = cfg['MODEL']['modelname'],
            modelparams     = cfg['MODEL'].get('params', dict()),
            strategy        = cfg['AL']['strategy'],
            n_start         = cfg['AL']['n_start'],
            n_end           = cfg['AL']['n_end'],
            n_query         = cfg['AL']['n_query'],
            n_subset        = cfg['AL']['n_subset'],
            trainset        = trainset,
            testset         = testset,
            img_size        = cfg['DATASET']['img_size'],
            num_classes     = cfg['DATASET']['num_classes'],
            batch_size      = cfg['DATASET']['batch_size'],
            test_batch_size = cfg['DATASET']['test_batch_size'],
            num_workers     = cfg['DATASET']['num_workers'],
            opt_name        = cfg['OPTIMIZER']['opt_name'],
            lr              = cfg['OPTIMIZER']['lr'],
            epochs          = cfg['TRAIN']['epochs'],
            log_interval    = cfg['TRAIN']['log_interval'],
            use_wandb       = cfg['TRAIN']['use_wandb'],
            savedir         = savedir,
            seed            = cfg['DEFAULT']['seed'],
            accelerator     = accelerator,
            cfg             = cfg
        )
    else:
        # make save directory
        savedir = os.path.join(cfg['DEFAULT']['savedir'], cfg['DATASET']['dataname'], cfg['MODEL']['modelname'], cfg['DEFAULT']['exp_name'])
        os.makedirs(savedir, exist_ok=True)
        
        # initialize wandb
        if cfg['TRAIN']['use_wandb']:
            wandb.init(name=cfg['DEFAULT']['exp_name'], project='Active Learning - Benchmark', entity='dsba-al-2023', config=cfg)  
        
        # run full supervised learning
        full_run(
            modelname       = cfg['MODEL']['modelname'],
            trainset        = trainset,
            testset         = testset,
            img_size        = cfg['DATASET']['img_size'],
            num_classes     = cfg['DATASET']['num_classes'],
            batch_size      = cfg['DATASET']['batch_size'],
            test_batch_size = cfg['DATASET']['test_batch_size'],
            num_workers     = cfg['DATASET']['num_workers'],
            opt_name        = cfg['OPTIMIZER']['opt_name'],
            lr              = cfg['OPTIMIZER']['lr'],
            epochs          = cfg['TRAIN']['epochs'],
            log_interval    = cfg['TRAIN']['log_interval'],
            use_wandb       = cfg['TRAIN']['use_wandb'],
            savedir         = savedir,
            seed            = cfg['DEFAULT']['seed'],
            accelerator     = accelerator
        )
    
    

def parser():
    parser = argparse.ArgumentParser(description='Active Learning - Benchmark')
    
    parser.add_argument('--yaml_config', type=str, default=None, help='default configuration file path')
    
    # defuault
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--seed', type=int, default=None, help='set seed')
    
    # DATASET
    parser.add_argument('--dataname', type=str, default=None, choices=['CIFAR10','CIFAR100','SVHN','Tiny_ImageNet_200'], help='data name')
    parser.add_argument('--datadir', type=str, default=None, help='data directory')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for trainset')
    parser.add_argument('--test_batch_size', type=int, default=None, help='batch size for testset')
    parser.add_argument('--num_workers', type=int, default=None, help='number of workers for preprocessing')
    
    # OPTIMIZER
    parser.add_argument('--opt_name', type=str, default=None, choices=['SGD','Adam'], help='optimizer name')
    parser.add_argument('--lr', type=float, default=None, help='learning rate for optimizer')
    
    # TRAIN
    parser.add_argument('--epochs', type=int, default=None, help='the number of epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=None, help='steps for gradients accumulation')
    parser.add_argument('--mixed_precision', type=str, default=None, choices=['fp16','bf16'], help='mixed precision')
    parser.add_argument('--log_interval', type=int, default=None, help='log interval')
    parser.add_argument('--no_wandb', action='store_true', help='use wandb')
    
    # Active Learning
    parser.add_argument('--n_start', type=int, default=None, help='number of samples for initial datasets')
    parser.add_argument('--n_query', type=int, default=None, help='number of query(budget or batch) for sampling')
    parser.add_argument('--n_end', type=int, default=None, help='number of samples to end active learning')
    parser.add_argument('--n_subset', type=int, default=None, help='number of samples for sub-sampling')
    
    # RESULT
    parser.add_argument('--savedir', type=str, default=None, help='directory to save result')

    args = parser.parse_args()
    args = vars(args)
    
    # check
    assert args['dataname'] != None, "dataname is not defined."
    
    # load config
    cfg = yaml.load(open(args['yaml_config'],'r'), Loader=yaml.FullLoader)
    
    # Update DATASET
    cfg['DATASET']['dataname'] = args['dataname']
    cfg['DATASET'].update(stats.datasets[args['dataname']])
    
    # update wandb
    cfg['TRAIN']['use_wandb'] = args['no_wandb'] == False
    
    # Update experiment name
    cfg['DEFAULT']['exp_name'] = cfg['AL']['strategy'] if 'AL' in cfg.keys() else 'Full'
    
    # Update arguments
    def update_value(cfg, group, key, value) -> dict:
        if key in cfg[group].keys() and value != None:
            if key == 'exp_name':
                cfg[group][key] = f"{cfg[group][key]}-{v}"    
            else:
                cfg[group][key] = value
        
        return cfg
        
    # update value
    for k, v in args.items():    
        for k_cfg in cfg.keys():
            cfg = update_value(cfg=cfg, group=k_cfg, key=k, value=v)

    return cfg    

if __name__=='__main__':

    # config
    cfg = parser()
    
    cfg_print = json.dumps(cfg, indent=4)
    print(cfg_print)
    
    # run
    run(cfg)