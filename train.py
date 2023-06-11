import logging
import wandb
import time
import os
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from collections import OrderedDict
from accelerate import Accelerator

from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, balanced_accuracy_score

from query_strategies import create_query_strategy
from models import create_model

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets, return_correct=False):
    # calculate accuracy
    preds = outputs.argmax(dim=1) 
    correct = targets.eq(preds).sum().item()
    
    if return_correct:
        return correct
    else:
        return correct/targets.size(0)


def calc_metrics(y_true: list, y_score: np.ndarray, y_pred: list) -> dict:
    # softmax
    y_score = torch.nn.functional.softmax(torch.FloatTensor(y_score), dim=1)
    
    # metrics
    auroc = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    bcr = balanced_accuracy_score(y_true, y_pred)

    return {
        'auroc'     : auroc, 
        'f1'        : f1, 
        'recall'    : recall, 
        'precision' : precision,
        'bcr'       : bcr
    }


def train(model, dataloader, criterion, optimizer, accelerator: Accelerator, log_interval: int) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    total_preds = []
    total_score = []
    total_targets = []
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        with accelerator.accumulate(model):
            data_time_m.update(time.time() - end)
            
            # predict
            outputs = model(inputs)
            loss = criterion(outputs, targets)    
            accelerator.backward(loss)

            # loss update
            optimizer.step()
            optimizer.zero_grad()
            losses_m.update(loss.item())

            # accuracy            
            acc_m.update(accuracy(outputs, targets), n=targets.size(0))
            
            # stack output
            total_preds.extend(outputs.argmax(dim=1).detach().cpu().tolist())
            total_score.extend(outputs.detach().cpu().tolist())
            total_targets.extend(targets.detach().cpu().tolist())
            
            # batch time
            batch_time_m.update(time.time() - end)
        
            if (idx+1) % accelerator.gradient_accumulation_steps == 0:
                if  ((idx+1) // accelerator.gradient_accumulation_steps) % log_interval == 0: 
                    _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                                'Acc: {acc.avg:.3%} '
                                'LR: {lr:.3e} '
                                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                (idx+1)//accelerator.gradient_accumulation_steps, 
                                len(dataloader)//accelerator.gradient_accumulation_steps, 
                                loss       = losses_m, 
                                acc        = acc_m, 
                                lr         = optimizer.param_groups[0]['lr'],
                                batch_time = batch_time_m,
                                rate       = inputs.size(0) / batch_time_m.val,
                                rate_avg   = inputs.size(0) / batch_time_m.avg,
                                data_time  = data_time_m))
    
            end = time.time()
    
    # calculate metrics
    metrics = calc_metrics(
        y_true  = total_targets,
        y_score = total_score,
        y_pred  = total_preds
    )
    
    metrics.update([('acc',acc_m.avg), ('loss',losses_m.avg)])
    
    # logging metrics
    _logger.info('\nTRAIN: Loss: %.3f | Acc: %.3f%% | BCR: %.3f%% | AUROC: %.3f%% | F1-Score: %.3f%% | Recall: %.3f%% | Precision: %.3f%%\n' % 
                 (metrics['loss'], 100.*metrics['acc'], 100.*metrics['bcr'], 100.*metrics['auroc'], 100.*metrics['f1'], 100.*metrics['recall'], 100.*metrics['precision']))
    
    return metrics
        
        
def test(model, dataloader, criterion, log_interval: int, name: str = 'TEST') -> dict:
    correct = 0
    total = 0
    total_loss = 0
    
    total_preds = []
    total_score = []
    total_targets = []
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            # predict
            outputs = model(inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            correct += accuracy(outputs, targets, return_correct=True)
            total += targets.size(0)
            
            # stack output
            total_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            total_score.extend(outputs.cpu().tolist())
            total_targets.extend(targets.cpu().tolist())
            
            if (idx+1) % log_interval == 0: 
                _logger.info('{0:s} [{1:d}/{2:d}]: Loss: {3:.3f} | Acc: {4:.3f}% [{5:d}/{6:d}]'.format(name, idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
    
    # calculate metrics
    metrics = calc_metrics(
        y_true  = total_targets,
        y_score = total_score,
        y_pred  = total_preds
    )
    
    metrics.update([('acc',correct/total), ('loss',total_loss/len(dataloader))])
    
    # logging metrics
    _logger.info('\n%s: Loss: %.3f | Acc: %.3f%% | BCR: %.3f%% | AUROC: %.3f%% | F1-Score: %.3f%% | Recall: %.3f%% | Precision: %.3f%%\n' % 
                 (name, metrics['loss'], 100.*metrics['acc'], 100.*metrics['bcr'], 100.*metrics['auroc'], 100.*metrics['f1'], 100.*metrics['recall'], 100.*metrics['precision']))
    
    return metrics
            
                
def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, accelerator: Accelerator,
    epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None, ckp_metric: str = None
) -> None:

    step = 0
    best_score = 0
    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train(
            model        = model, 
            dataloader   = trainloader, 
            criterion    = criterion, 
            optimizer    = optimizer, 
            accelerator  = accelerator, 
            log_interval = log_interval
        )
        
        eval_metrics = test(
            model        = model, 
            dataloader   = testloader, 
            criterion    = criterion, 
            log_interval = log_interval,
            name         = 'VALID'
        )

        if scheduler:
            scheduler.step()

        # wandb
        if use_wandb:
            metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
            metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
            wandb.log(metrics, step=step)

        step += 1
        
        # checkpoint - save best results and model weights
        if ckp_metric:
            ckp_cond = (best_score > eval_metrics[ckp_metric]) if ckp_metric == 'loss' else (best_score < eval_metrics[ckp_metric])
            if savedir and ckp_cond:
                best_score = eval_metrics[ckp_metric]
                state = {'best_step':step}
                state.update(eval_metrics)
                json.dump(state, open(os.path.join(savedir, f'results_seed{seed}_best.json'), 'w'), indent='\t')
                torch.save(model.state_dict(), os.path.join(savedir, f'model_seed{seed}_best.pt'))
    
    return eval_metrics


def full_run(
    modelname: str, pretrained: bool,
    trainset, validset, testset,
    img_size: int, num_classes: int, batch_size: int, test_batch_size: int, num_workers: int, 
    opt_name: str, lr: float, 
    epochs: int, log_interval: int, use_wandb: bool, savedir: str, seed: int, accelerator: Accelerator, ckp_metric: str = None):
    
    # logging
    _logger.info('Full Supervised Learning, [total samples] {}'.format(len(trainset)))

    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers
    )
    
    # define test dataloader
    validloader = DataLoader(
        dataset     = validset,
        batch_size  = test_batch_size,
        shuffle     = False,
        num_workers = num_workers
    )
    
    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = test_batch_size,
        shuffle     = False,
        num_workers = num_workers
    )

    # load model
    model = create_model(
        modelname   = modelname, 
        num_classes = num_classes, 
        img_size    = img_size, 
        pretrained  = pretrained
    )
    
    # optimizer
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name](model.parameters(), lr=lr)

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=0.00001)

    # criterion 
    criterion = torch.nn.CrossEntropyLoss()
    
    # prepraring accelerator
    model, optimizer, trainloader, validloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, validloader, testloader, scheduler
    )

    # fitting model
    _ = fit(
        model        = model, 
        trainloader  = trainloader, 
        testloader   = validloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        accelerator  = accelerator,
        epochs       = epochs, 
        use_wandb    = use_wandb,
        log_interval = log_interval,
        savedir      = savedir if validset != testset else None,
        seed         = seed if validset != testset else None,
        ckp_metric   = ckp_metric if validset != testset else None
    )
    
    # save model
    torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{seed}.pt"))

    # test results
    test_results = test(
        model        = model, 
        dataloader   = testloader, 
        criterion    = criterion, 
        log_interval = log_interval
    )

    # save result
    json.dump(test_results, open(os.path.join(savedir, f'results_seed{seed}.json'), 'w'), indent='\t')


def al_run(
    exp_name: str, modelname: str, pretrained: bool,
    strategy: str, n_start: int, n_end: int, n_query: int, n_subset: int, 
    trainset, validset, testset,
    img_size: int, num_classes: int, batch_size: int, test_batch_size: int, num_workers: int, 
    opt_name: str, lr: float, 
    epochs: int, log_interval: int, use_wandb: bool, savedir: str, seed: int, accelerator: Accelerator, ckp_metric: str = None, cfg: dict = None):
    
    assert cfg != None if use_wandb else True, 'If you use wandb, configs should be exist.'
    
    # set active learning arguments
    nb_round = (n_end - n_start)/n_query
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # logging
    _logger.info('[total samples] {}, [initial samples] {} [qeury samples] {} [end samples] {} [total round] {}'.format(
        len(trainset), n_start, n_query, n_end, nb_round))
    
    # inital sampling labeling
    sample_idx = np.arange(len(trainset))
    np.random.shuffle(sample_idx)
    
    labeled_idx = np.zeros_like(sample_idx, dtype=bool)
    labeled_idx[sample_idx[:n_start]] = True
    
    # select strategy    
    strategy = create_query_strategy(
        strategy_name = strategy, 
        model         = create_model(modelname=modelname, num_classes=num_classes, img_size=img_size, pretrained=pretrained),
        dataset       = trainset, 
        labeled_idx   = labeled_idx, 
        n_query       = n_query, 
        batch_size    = batch_size, 
        num_workers   = num_workers,
        params        = cfg.MODEL.get('params', dict())
    )
    
    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = batch_size,
        sampler     = SubsetRandomSampler(indices=np.where(labeled_idx==True)[0]),
        num_workers = num_workers
    )
    
    # define test dataloader
    validloader = DataLoader(
        dataset     = validset,
        batch_size  = test_batch_size,
        shuffle     = False,
        num_workers = num_workers
    )
    
    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = test_batch_size,
        shuffle     = False,
        num_workers = num_workers
    )
    
    # define log dataframe
    log_df = pd.DataFrame(
        columns=['round', 'auroc', 'f1', 'recall', 'precision', 'bcr', 'acc', 'loss']
    )
    
    # query log dataframe
    query_log_df = pd.DataFrame({'idx': range(len(labeled_idx))})
    query_log_df['query_round'] = None
    query_log_df.loc[labeled_idx, 'query_round'] = 'round0'
    
    # run
    for r in range(nb_round+1):
        
        if r != 0:    
            # query sampling    
            query_idx = strategy.query(model, n_subset=n_subset)
            trainloader = strategy.update(query_idx)
            
            # save query index
            query_log_df.loc[query_idx, 'query_round'] = f'round{r}'
            query_log_df.to_csv(os.path.join(savedir, 'query_log.csv'), index=False)
            
        # logging
        _logger.info('[Round {}/{}] training samples: {}'.format(r, nb_round, sum(strategy.labeled_idx)))
        
        # build Model
        model = strategy.init_model()
        
        # optimizer
        optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name](model.parameters(), lr=lr)

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=0.00001)
        
        # prepraring accelerator
        model, optimizer, trainloader, validloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, validloader, testloader, scheduler
        )
        
        # initialize wandb
        if use_wandb:
            wandb.init(name=f'{exp_name}_round{r}', project=cfg.wandb.project_name, entity=cfg.wandb.entity, config=cfg)        

        # fitting model
        _ = fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = validloader, 
            criterion    = strategy.loss_fn, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            epochs       = epochs, 
            use_wandb    = use_wandb,
            log_interval = log_interval,
            savedir      = savedir if validset != testset else None,
            seed         = seed if validset != testset else None,
            ckp_metric   = ckp_metric if validset != testset else None
        )
        
        # save model
        torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{seed}.pt"))

        # test results
        test_results = test(
            model        = model, 
            dataloader   = testloader, 
            criterion    = strategy.loss_fn, 
            log_interval = log_interval
        )

        # save results
        log_metrics = {'round':r}
        log_metrics.update([(k, v) for k, v in test_results.items()])
        log_df = log_df.append(log_metrics, ignore_index=True)
        
        log_df.round(4).to_csv(
            os.path.join(savedir, f"round{nb_round}-seed{seed}.csv"),
            index=False
        )   
        
        _logger.info('append result [shape: {}]'.format(log_df.shape))
        
        wandb.finish()