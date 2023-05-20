import numpy as np
import pandas as pd
import os
import random
import wandb
import torch
import argparse
import yaml
import logging
from glob import glob

from torch.utils.data import DataLoader
from train import fit, test
from datasets import create_dataset
from query_strategies import create_query_strategy
from models import create_model
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


def update_label_info(trainset, exp_name: str, round: int):
    
    # add previous labels of query to label infomation of trainset
    if round > 0:
        previous_results = pd.concat([pd.read_csv(f) for f in glob(os.path.join(exp_name,'*/query_list.csv'))])
        print(previous_results.shape)
        # update
        query_imgs = trainset.label_info.img_path.isin(previous_results.img_path)
        trainset.label_info.loc[query_imgs, 'label'] = previous_results['label'].tolist()
        trainset.label_info.loc[query_imgs, 'labeled_yn'] = previous_results['labeled_yn'].tolist()

    trainset.data_info = trainset.label_info[trainset.label_info.labeled_yn==True]
    labeled_idx = np.array(trainset.label_info.labeled_yn.tolist())
    
    return trainset, labeled_idx


def save_results(exp_name, round, acc, n_label):
    # load previous log dataframe
    if round == 0:
        log_df = pd.DataFrame(columns=['round','acc','n_label'])
    else:
        log_df = pd.read_csv(os.path.join(exp_name, 'test_results.csv'))
        
    # append results
    log_df = log_df.append({
        'round'   : round,
        'acc'     : acc,
        'n_label' : n_label
    }, ignore_index=True)
    
    # save
    log_df.to_csv(
        os.path.join(exp_name, 'test_results.csv'),
        index=False
    )    
    
    _logger.info('append result [shape: {}]'.format(log_df.shape))
    

def run(cfg):

    # define exp name
    exp_name = os.path.join(
        cfg['RESULT']['savedir'], 
        cfg['DATASET']['dataname'], 
        cfg['MODEL']['modelname'],
        f"{cfg['AL']['strategy']}-n_query{cfg['AL']['n_query']}"
    )
    
    # check round
    if not os.path.isdir(exp_name):
        round = 0
    else:
        round = len(glob(os.path.join(exp_name, 'round*')))
    
    # make save directory
    savedir = os.path.join(exp_name, f'round{round}')
    os.makedirs(savedir, exist_ok=True)

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg['TRAIN']['grad_accum_steps'],
        mixed_precision             = cfg['TRAIN']['mixed_precision']
    )

    setup_default_logging(log_path=os.path.join(savedir, 'log.txt'))
    torch_seed(cfg['SEED'])

    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # load dataset
    trainset, validset, testset = create_dataset(
        datadir  = cfg['DATASET']['datadir'], 
        dataname = cfg['DATASET']['dataname'],
        img_size = cfg['DATASET']['img_size'],
        mean     = cfg['DATASET']['mean'],
        std      = cfg['DATASET']['std']
    )
    
    # add query labels annotated from previous rounds
    trainset, labeled_idx = update_label_info(
        trainset = trainset,
        exp_name = exp_name,
        round    = round
    )
        
    # select strategy
    strategy = create_query_strategy(
        strategy_name = cfg['AL']['strategy'], 
        model         = create_model(modelname=cfg['MODEL']['modelname'], num_classes=cfg['DATASET']['num_classesl'], img_size=cfg['DATASET']['img_size'], pretrained=cfg['MODEL']['pretrained']),
        dataset       = trainset, 
        labeled_idx   = labeled_idx, 
        n_query       = cfg['AL']['n_query'], 
        batch_size    = cfg['DATASET']['batch_size'], 
        num_workers   = cfg['DATASET']['num_workers']
    )
    
    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = cfg['DATASET']['batch_size'],
        shuffle     = True, 
        num_workers = cfg['DATASET']['num_workers']
    )
    
    # define test dataloader
    validloader = DataLoader(
        dataset     = validset,
        batch_size  = cfg['DATASET']['test_batch_size'],
        shuffle     = False,
        num_workers = cfg['DATASET']['num_workers']
    )
    
    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = cfg['DATASET']['test_batch_size'],
        shuffle     = False,
        num_workers = cfg['DATASET']['num_workers']
    )
    
    # build model
    model = strategy.init_model()
    
    # optimizer
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['OPTIMIZER']['opt_name']](model.parameters(), lr=cfg['OPTIMIZER']['lr'])

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['TRAIN']['epochs'], T_mult=1, eta_min=0.00001)
    
    # prepraring accelerator
    model, optimizer, trainloader, validloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, validloader, testloader, scheduler
    )
    
    # initialize wandb
    if cfg['TRAIN']['use_wandb']:
        wandb.init(name=exp_name+f'_round{r}', project='Active Learning - Round', config=cfg)        

    # logging
    _logger.info('[Round: {}] training samples: {}'.format(round, sum(labeled_idx)))


    # fitting model
    _ = fit(
        model        = model, 
        trainloader  = trainloader, 
        testloader   = validloader, 
        criterion    = strategy.loss_fn,
        optimizer    = optimizer, 
        scheduler    = scheduler,
        accelerator  = accelerator,
        epochs       = cfg['TRAIN']['epochs'], 
        use_wandb    = cfg['TRAIN']['use_wandb'],
        log_interval = cfg['TRAIN']['log_interval'],
        savedir      = savedir
    )
    
    _logger.info('\n[TESTSET EVALUATION]')
    # test
    test_results = test(
        model        = model, 
        dataloader   = testloader, 
        criterion    = strategy.loss_fn, 
        log_interval = cfg['TRAIN']['log_interval']
    )
    
    # change data information into unlabeled data
    trainset.data_info = trainset.label_info[trainset.label_info.labeled_yn==False]
    
    # query
    query_idx = strategy.query(model, n_subset=cfg['AL']['n_subset'])
    
    # save query list
    query_df = trainset.label_info.iloc[query_idx]
    query_df.to_csv(os.path.join(savedir,'query_list.csv'), index=False)

    # save results
    save_results(
        exp_name = exp_name,
        round = round,
        acc = test_results['acc'],
        n_label = len(trainset.label_info[trainset.label_info.labeled_yn==True])
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Active Learning - Round')
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')    

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    run(cfg)