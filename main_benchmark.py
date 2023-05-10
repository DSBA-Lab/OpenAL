import numpy as np
import pandas as pd
import os
import random
import wandb
import torch
import argparse
import yaml
import logging

from torch.utils.data import DataLoader, SubsetRandomSampler
from train import fit
from datasets import stats
from query_strategies import create_query_strategy
from models import *
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

    # make save directory
    al_name = f"total_{cfg['AL']['n_end']}-init_{cfg['AL']['n_start']}-query_{cfg['AL']['n_query']}"
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['dataname'], cfg['MODEL']['modelname'], cfg['EXP_NAME'], al_name)
    os.makedirs(savedir, exist_ok=True)

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg['TRAIN']['grad_accum_steps'],
        mixed_precision             = cfg['TRAIN']['mixed_precision']
    )

    setup_default_logging()
    torch_seed(cfg['SEED'])

    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # load dataset
    trainset, testset = __import__('datasets').__dict__[f"load_{cfg['DATASET']['dataname'].lower()}"](
        datadir            = cfg['DATASET']['datadir'], 
        img_size           = cfg['DATASET']['img_size'],
        mean               = cfg['DATASET']['mean'], 
        std                = cfg['DATASET']['std']
    )
    
    # set active learning arguments
    nb_round = (cfg['AL']['n_end'] - cfg['AL']['n_start'])/cfg['AL']['n_query']
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # logging
    _logger.info('[total samples] {}, [initial samples] {} [qeury samples] {} [end samples] {} [total round] {}'.format(
        len(trainset), cfg['AL']['n_start'], cfg['AL']['n_query'], cfg['AL']['n_end'], nb_round))
    
    # inital sampling labeling
    sample_idx = np.arange(len(trainset))
    np.random.shuffle(sample_idx)
    
    labeled_idx = np.zeros_like(sample_idx, dtype=bool)
    labeled_idx[sample_idx[:cfg['AL']['n_start']]] = True
    
    # select strategy    
    strategy = create_query_strategy(
        strategy_name = cfg['AL']['strategy'], 
        model         = __import__('models').__dict__[cfg['MODEL']['modelname']](num_classes=cfg['DATASET']['num_classes']),
        dataset       = trainset, 
        labeled_idx   = labeled_idx, 
        n_query       = cfg['AL']['n_query'], 
        batch_size    = cfg['DATASET']['batch_size'], 
        num_workers   = cfg['DATASET']['num_workers'],
        params        = cfg['MODEL'].get('params', dict())
    )
    
    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = cfg['DATASET']['batch_size'],
        sampler     = SubsetRandomSampler(indices=np.where(labeled_idx==True)[0]),
        num_workers = cfg['DATASET']['num_workers']
    )
    
     # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = cfg['DATASET']['test_batch_size'],
        shuffle     = False,
        num_workers = cfg['DATASET']['num_workers']
    )
    
    # define log df
    log_df = pd.DataFrame(
        columns=['round','acc']
    )
    
    # run
    for r in range(nb_round+1):
        
        if r != 0:    
            # query sampling    
            query_idx = strategy.query(model, n_subset=cfg['AL']['n_subset'])
            trainloader = strategy.update(query_idx)
            
        # logging
        _logger.info('[Round {}/{}] training samples: {}'.format(r, nb_round, sum(strategy.labeled_idx)))
        
        # build Model
        model = strategy.init_model()
        
        # optimizer
        optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['OPTIMIZER']['opt_name']](model.parameters(), lr=cfg['OPTIMIZER']['lr'])

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['TRAIN']['epochs'], T_mult=1, eta_min=0.00001)
        
        # prepraring accelerator
        model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, testloader, scheduler
        )
        
        # initialize wandb
        if cfg['TRAIN']['use_wandb']:
            wandb.init(name=cfg['EXP_NAME']+f'_round{r}', project='Active Learning - Benchmark', entity='dsba-al-2023', config=cfg)        

        # fitting model
        test_results = fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = testloader, 
            criterion    = strategy.loss_fn, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            epochs       = cfg['TRAIN']['epochs'], 
            use_wandb    = cfg['TRAIN']['use_wandb'],
            log_interval = cfg['TRAIN']['log_interval'],
        )
        
        # save model
        torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{cfg['SEED']}.pt"))

        # save results
        log_df = log_df.append({
            'round' : r,
            'acc'   : test_results['acc']
        }, ignore_index=True)
        
        log_df.to_csv(
            os.path.join(savedir, f"round_{nb_round}-seed{cfg['SEED']}.csv"),
            index=False
        )    
        
        _logger.info('append result [shape: {}]'.format(log_df.shape))
        
        wandb.finish()
    

def parser(cfg):
    parser = argparse.ArgumentParser(description='Active Learning - Benchmark')
    
    # defuault
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--seed', type=int, default=1, help='set seed')
    
    # DATASET
    parser.add_argument('--dataname', type=str, default='CIFAR10', choices=['CIFAR10','CIFAR100','SVHN','Tiny-ImageNet-200'], help='data name')
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
    parser.add_argument('--use_wandb', type=bool, default=None, help='use wandb')
    
    # Active Learning
    parser.add_argument('--n_start', type=int, default=None, help='number of samples for initial datasets')
    parser.add_argument('--n_query', type=int, default=None, help='number of query(budget or batch) for sampling')
    parser.add_argument('--n_end', type=int, default=None, help='number of samples to end active learning')
    parser.add_argument('--n_subset', type=int, default=None, help='number of samples for sub-sampling')
    
    # RESULT
    parser.add_argument('--savedir', type=str, default=None, help='directory to save result')

    args = parser.parse_args()
    args = vars(args)
    
    # Update DATASET
    cfg['DATASET'].update(stats.datasets[args['dataname']])
    
    # Update experiment name
    cfg['DEFAULT']['exp_name'] = cfg['AL']['strategy']
    
    # Update arguments
    def update_value(cfg, group, key, value) -> dict:
        if key in cfg[group].keys() and value:
            if key == 'exp_name':
                cfg[group][key] = f"{cfg[group][key]}-{v}"    
            else:
                cfg[group][key] = value
        
        return cfg
        
    # update value
    for k, v in args.items():    
        print(k)
        for k_cfg in cfg.keys():
            cfg = update_value(cfg=cfg, group=k_cfg, key=k, value=v)

    return cfg    

if __name__=='__main__':

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    cfg = parser(cfg)
    
    run(cfg)