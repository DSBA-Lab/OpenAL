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
from datasets import create_dataset_benchmark
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
    trainset, testset = create_dataset_benchmark(
        datadir  = cfg['DATASET']['datadir'], 
        dataname = cfg['DATASET']['dataname'],
        img_size = cfg['DATASET']['img_size']
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
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Active Learning - Benchmark')
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')    
    parser.add_argument('--seed', type=int, default=None, help='set seed')

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    
    if args.seed != None:
        cfg['SEED'] = args.seed

    run(cfg)