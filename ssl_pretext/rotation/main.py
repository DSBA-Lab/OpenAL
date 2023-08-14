import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, '../../'))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from utils import torch_seed
from train import fit
from datasets import create_dataset
from models import create_model
from log import setup_default_logging
from arguments import parser_ssl
from ssl_pretext.rotation import RotationDataset

_logger = logging.getLogger('train')

def get_target_from_dataset(dataset):
    # if class name is ALDataset
    if dataset.__class__.__name__ == "ALDataset":
        targets = dataset.data_info.label.tolist()
    else:
       # attribution name list in benchmark dataset class
        target_attrs = ['targets', 'labels'] # TODO: if target attribution name is added, append in this line.

        # iterativly check attribution name if not False else break
        for attr in target_attrs:
            targets = getattr(dataset, attr, False)
            if targets is not False:
                break

    return targets

def make_batch(model, dataloader, criterion, savedir: str, log_interval: int):    
    model.eval()
    total_loss, correct, total = 0, 0, 0
    losses = []
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            # inputs: (B x 4 x C x H x W)
            # targets: (B x 4)

            # inference
            outputs = [model(x) for x in inputs]
        
            # loss
            loss = [criterion(x,y).item() for x,y in zip(outputs, targets)]
            total_loss += np.mean(loss)
            losses.extend(loss)
            
            # targets
            targets = torch.stack([t[0] for t in targets]) # the target only considers the 0 th class
            total += targets.size(0)
            
            # accuracy of original images
            predicted = torch.stack([o[0].argmax() for o in outputs]) # prediction of original image
            correct += predicted.eq(targets).sum().item()
            
            if (idx+1) % log_interval == 0: 
                _logger.info('Make Batch [{0:d}/{1:d}]: Loss: {2:.3f} | Acc: {3:.3f}% [{4:d}/{5:d}]'.format(idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
                
    
    # make dataframe for batch
    df_batch = pd.DataFrame({'idx': range(len(losses)), 'loss': losses})
    df_batch['target'] = get_target_from_dataset(dataloader.dataset.dataset)
    
    # sorting by loss
    df_batch = df_batch.sort_values('loss', ascending=False)
    
    # save batch
    df_batch.to_csv(os.path.join(savedir, 'batch.csv'), index=False)


def batch_stack(batch):
    '''
    Arg:
    - batch (list): batch samples. input in batch is (4, C, H, W). ex) batch size = 2, [(input 1, target 1), (input 2, target 2)].
    
    Return:
    - inputs (torch.Tensor): stacked inputs. ex) (Bx4, C, H, W)
    - targets (torch.Tensor): stacked targets. ex) (Bx4,)
    
    '''
    inputs = torch.cat([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    return inputs, targets


def run(cfg):
    # save directory
    savedir = os.path.join(cfg.DEFAULT.savedir, cfg.DATASET.dataname, cfg.MODEL.modelname, cfg.DEFAULT.exp_name)
    assert not os.path.isdir(savedir), f'{savedir} already exists'
    os.makedirs(savedir)
    
    # save config
    OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
    
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )
    
    # define logging and seed
    setup_default_logging()
    torch_seed(cfg.DEFAULT.seed)
    
    # load dataset
    if f"load_{cfg.DATASET.dataname.lower()}" in __import__('datasets').__dict__.keys():
        trainset, validset = __import__('datasets').__dict__[f"load_{cfg.DATASET.dataname.lower()}"](
            datadir  = cfg.DATASET.datadir, 
            img_size = cfg.DATASET.img_size,
            mean     = cfg.DATASET.mean, 
            std      = cfg.DATASET.std,
            aug_info = cfg.DATASET.aug_info
        )
    else:
        trainset, validset, _ = create_dataset(
            datadir  = cfg.DATASET.datadir, 
            dataname = cfg.DATASET.dataname,
            img_size = cfg.DATASET.img_size,
            mean     = cfg.DATASET.mean,
            std      = cfg.DATASET.std,
            aug_info = cfg.DATASET.aug_info,
            seed     = cfg.DATASET.seed
        )
        
    # build rotation dataset
    '''
    if is_train is True, rotation angle is randomly selected
    if is_train is False, all rotation angles are returned
    '''
    trainset_rot = RotationDataset(dataset=trainset, is_train=True)  # (C, H, W)
    validset_rot = RotationDataset(dataset=validset, is_train=False) # (4, C, H, W), 4 is the number of rotation angles
    
    # dataloader
    trainloader = DataLoader(
        dataset     = trainset_rot,
        batch_size  = cfg.DATASET.batch_size,
        shuffle     = True,
        num_workers = cfg.DATASET.num_workers
    )
    
    validloader = DataLoader(
        dataset     = validset_rot,
        batch_size  = cfg.DATASET.test_batch_size,
        shuffle     = False,
        num_workers = cfg.DATASET.num_workers,
        collate_fn  = batch_stack # collate function for validloader
    )
    
    # build dataset for saving batch
    batchset = RotationDataset(dataset=trainset, is_train=False)
    batchloader = DataLoader(
        dataset     = batchset,
        batch_size  = cfg.DATASET.test_batch_size,
        shuffle     = False,
        num_workers = cfg.DATASET.num_workers
    )
    
    # load model
    cfg.DATASET.num_classes = 4 # 4 is the number of rotation angles
    model = create_model(
        modelname   = cfg.MODEL.modelname, 
        num_classes = cfg.DATASET.num_classes, 
        img_size    = cfg.DATASET.img_size, 
        pretrained  = cfg.MODEL.pretrained
    )
    
    # criterion
    criterion = nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg.OPTIMIZER.opt_name](model.parameters(), lr=cfg.OPTIMIZER.lr, **cfg.OPTIMIZER.get('params',{}))
    
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])
    
    # prepraring accelerator
    model, optimizer, trainloader, validloader, batchloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, validloader, batchloader, scheduler
    )
    
    # initialize wandb
    if cfg.TRAIN.wandb.use:
        wandb.init(name=cfg.DEFAULT.exp_name, project=cfg.TRAIN.wandb.project_name, entity=cfg.TRAIN.wandb.entity, config=OmegaConf.to_container(cfg))
    
    # fitting model
    fit(
        model        = model, 
        trainloader  = trainloader, 
        testloader   = validloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        accelerator  = accelerator,
        epochs       = cfg.TRAIN.epochs, 
        use_wandb    = cfg.TRAIN.wandb.use,
        log_interval = cfg.TRAIN.log_interval,
        savedir      = savedir,
        seed         = cfg.DEFAULT.seed,
        ckp_metric   = cfg.TRAIN.ckp_metric
    )
    
    # load checkpoint
    state_dict = torch.load(os.path.join(savedir, f'model_seed{cfg.DEFAULT.seed}_best.pt'))
    model.load_state_dict(state_dict)
    
    # make batch
    make_batch(
        model        = model,
        dataloader   = batchloader,
        criterion    = criterion,  
        savedir      = savedir,
        log_interval = cfg.TRAIN.log_interval
    )

        
if __name__ == "__main__":
    # config
    cfg = parser_ssl()
    run(cfg)
    