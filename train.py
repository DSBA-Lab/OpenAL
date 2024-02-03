import logging
import wandb
import time
import os
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from accelerate import Accelerator

from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, \
                            balanced_accuracy_score, classification_report, confusion_matrix

from query_strategies import create_query_strategy, create_is_labeled, \
                             create_id_ood_targets, create_is_labeled_unlabeled, \
                             create_id_testloader, torch_seed, \
                             create_scheduler, create_optimizer
from query_strategies.utils import TrainIterableDataset
from query_strategies.prompt_ensemble import PromptEnsemble
from models import create_model
from query_strategies.utils import NoIndent, MyEncoder
from omegaconf import OmegaConf

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


class PromptLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, loss_weight: float = 1., weight=None, size_average=None, ignore_index=-100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0):
        super(PromptLoss, self).__init__(
            weight          = weight, 
            size_average    = size_average, 
            ignore_index    = ignore_index, 
            reduce          = reduce, 
            reduction       = reduction, 
            label_smoothing = label_smoothing
        )
        
        self.loss_weight = loss_weight
        
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        ce_loss = torch.nn.functional.cross_entropy(input['logits'], target, weight=self.weight,
                                                        ignore_index=self.ignore_index, reduction=self.reduction,
                                                        label_smoothing=self.label_smoothing)
        
        prompt_ce_loss = torch.nn.functional.cross_entropy(input['prompt_logits'], target, weight=self.weight,
                                                     ignore_index=self.ignore_index, reduction=self.reduction,
                                                     label_smoothing=self.label_smoothing)
        
        loss = ce_loss + self.loss_weight * prompt_ce_loss
        
        return loss

def accuracy(outputs, targets, return_correct=False):
    # calculate accuracy
    preds = outputs.argmax(dim=1) 
    correct = targets.eq(preds).sum().item()
    
    if return_correct:
        return correct
    else:
        return correct/targets.size(0)


def create_criterion(name, params):
    if name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss(**params)
    elif name == 'PromptLoss':
        return PromptLoss(**params)

def calc_metrics(y_true: list, y_score: np.ndarray, y_pred: list, return_per_class: bool = False) -> dict:
    # softmax
    y_score = torch.nn.functional.softmax(torch.FloatTensor(y_score), dim=1)
    
    # metrics
    auroc = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0.0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0.0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0.0)
    bcr = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        'auroc'     : auroc, 
        'f1'        : f1, 
        'recall'    : recall, 
        'precision' : precision,
        'bcr'       : bcr
    }

    if return_per_class:
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # merics per class
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0.0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0.0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0.0)
        acc_per_class = cm.diagonal() / cm.sum(axis=1)
    
        metrics.update({
            'per_class':{
                'cm': [NoIndent(elem) for elem in cm.tolist()],
                'f1': f1_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'precision': precision_per_class.tolist(),
                'acc': acc_per_class.tolist()
            }
        })

    return metrics


def get_metrics(metrics: dict, metrics_log: str, targets: list, scores: list, preds: list, return_per_class: str = False):
    metrics.update(calc_metrics(
        y_true           = targets,
        y_score          = scores,
        y_pred           = preds,
        return_per_class = return_per_class
    ))
    metrics_log += ' | BCR: %.3f%% | AUROC: %.3f%% | F1-Score: %.3f%% | Recall: %.3f%% | Precision: %.3f%%\n' % \
                    (100.*metrics['bcr'], 100.*metrics['auroc'], 100.*metrics['f1'], 100.*metrics['recall'], 100.*metrics['precision'])

    # classification report
    _logger.info(classification_report(y_true=targets, y_pred=preds, digits=4, zero_division=0.0))
    
    return metrics, metrics_log



def train(model, dataloader, criterion, optimizer, accelerator: Accelerator, log_interval: int, query_model = None, **train_params) -> dict:   
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
    
    steps_per_epoch = train_params.get('steps_per_epoch') if train_params.get('steps_per_epoch') else len(dataloader)
    
    step = 0
    for idx, (inputs, targets) in enumerate(dataloader):
        with accelerator.accumulate(model):
            data_time_m.update(time.time() - end)
            
            # predict
            if query_model != None:
                with torch.no_grad():
                    cls_features = query_model.forward_features(inputs)[:,0]
                outputs = model(inputs, cls_features=cls_features)
            else:
                outputs = model(inputs)
            
            # detach LPM for learning loss
            if train_params.get('is_detach_lpm', False):
                for k, v in model.layer_outputs.items():
                    model.layer_outputs[k] = v.detach()

            # calc loss
            if criterion.__class__.__name__ == 'CrossEntropyLoss' and isinstance(outputs, dict):
                outputs = outputs['logits']
                
            loss = criterion(outputs, targets)    
            accelerator.backward(loss)

            # loss update
            optimizer.step()
            optimizer.zero_grad()
            losses_m.update(loss.item())

            # accuracy 
            if isinstance(outputs, dict):
                outputs = outputs['logits']           
            acc_m.update(accuracy(outputs, targets), n=targets.size(0))
            
            # stack output
            total_preds.extend(outputs.argmax(dim=1).detach().cpu().tolist())
            total_score.extend(outputs.detach().cpu().tolist())
            total_targets.extend(targets.detach().cpu().tolist())
            
            # batch time
            batch_time_m.update(time.time() - end)
        
            if (step+1) % accelerator.gradient_accumulation_steps == 0:
                if ((step+1) // accelerator.gradient_accumulation_steps) % log_interval == 0: 
                    _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                                'Acc: {acc.avg:.3%} '
                                'LR: {lr:.3e} '
                                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                (step+1)//accelerator.gradient_accumulation_steps, 
                                steps_per_epoch//accelerator.gradient_accumulation_steps, 
                                loss       = losses_m, 
                                acc        = acc_m, 
                                lr         = optimizer.param_groups[0]['lr'],
                                batch_time = batch_time_m,
                                rate       = inputs.size(0) / batch_time_m.val,
                                rate_avg   = inputs.size(0) / batch_time_m.avg,
                                data_time  = data_time_m))
    
            end = time.time()
            
            step += 1
            
            if step == steps_per_epoch:
                break

    # calculate metrics
    metrics = {}
    metrics.update([('acc',acc_m.avg), ('loss',losses_m.avg)])
    metrics_log = '\nTRAIN: Loss: %.3f | Acc: %.3f%%' % (metrics['loss'], 100.*metrics['acc'])
    
    if not train_params.get('metrics_off', False):
        metrics, metrics_log = get_metrics(
            metrics     = metrics, 
            metrics_log = metrics_log, 
            targets     = total_targets, 
            scores      = total_score, 
            preds       = total_preds, 
        )
    
    # logging metrics
    _logger.info(metrics_log)
        
    return metrics

        
def test(model, dataloader, criterion, log_interval: int, name: str = 'TEST', return_per_class: bool = False, query_model = None) -> dict:
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
            if query_model != None:
                cls_features = query_model.forward_features(inputs)[:,0]
                outputs = model(inputs, cls_features=cls_features)
            else:
                outputs = model(inputs)
            
            # loss
            if criterion.__class__.__name__ == 'CrossEntropyLoss' and isinstance(outputs, dict):
                outputs = outputs['logits']
            loss = criterion(outputs, targets)
            
            # total loss and acc
            if isinstance(outputs, dict):
                outputs = outputs['logits']
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
    metrics = {}
    metrics.update([('acc',correct/total), ('loss',total_loss/len(dataloader))])
    metrics_log = '\n%s: Loss: %.3f | Acc: %.3f%%' % (name, metrics['loss'], 100.*metrics['acc'])

    metrics, metrics_log = get_metrics(
        metrics          = metrics, 
        metrics_log      = metrics_log, 
        targets          = total_targets, 
        scores           = total_score, 
        preds            = total_preds, 
        return_per_class = return_per_class
    )
    
    # logging metrics
    _logger.info(metrics_log)
    
    return metrics
            
                
def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, accelerator: Accelerator,
    epochs: int, use_wandb: bool, log_interval: int, query_model = None, seed: int = 0, **train_params
) -> None:

    step = 0
    
    torch_seed(seed)    
    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        
        # for learning loss
        if 'detach_epoch_ratio' in train_params.keys():
            is_detach_lpm = True if epoch > int(epochs * train_params['detach_epoch_ratio']) else False
            train_params['is_detach_lpm'] = is_detach_lpm
        
        train_metrics = train(
            model        = model, 
            query_model  = query_model,
            dataloader   = trainloader, 
            criterion    = criterion, 
            optimizer    = optimizer, 
            accelerator  = accelerator, 
            log_interval = log_interval,
            **train_params
        )
        
        if testloader != None:
            eval_metrics = test(
                model        = model, 
                query_model  = query_model,
                dataloader   = testloader, 
                criterion    = criterion, 
                log_interval = log_interval,
                name         = 'VALID'
            )

        # wandb
        if use_wandb:
            metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
            metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
            
            if testloader != None:
                metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
                
            wandb.log(metrics, step=step)
        
        step += 1
        
        # update scheduler  
        if scheduler:
            scheduler.step()
        

def full_run(
    cfg: dict, trainset, validset, testset, savedir: str, accelerator: Accelerator):
    
    # logging
    _logger.info('Full Supervised Learning, [total samples] {}'.format(len(trainset)))

    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = cfg.DATASET.batch_size,
        shuffle     = True,
        num_workers = cfg.DATASET.num_workers,
        pin_memory  = True
    )
    
    if cfg.TRAIN.get('params'):
        if cfg.TRAIN.params.get('steps_per_epoch'):
            trainloader = DataLoader(
                dataset     = TrainIterableDataset(dataset=trainset),
                batch_size  = cfg.DATASET.batch_size,
                num_workers = cfg.DATASET.num_workers,
                pin_memory  = True
            )   
    
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = cfg.DATASET.batch_size,
        shuffle     = True,
        num_workers = cfg.DATASET.num_workers
    )
    
    # define test dataloader
    validloader = DataLoader(
        dataset     = validset,
        batch_size  = cfg.DATASET.test_batch_size,
        shuffle     = False,
        num_workers = cfg.DATASET.num_workers
    )
    
    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = cfg.DATASET.test_batch_size,
        shuffle     = False,
        num_workers = cfg.DATASET.num_workers
    )

    # load model
    query_model, model = create_model(
        modelname   = cfg.MODEL.name, 
        num_classes = cfg.DATASET.num_classes, 
        pretrained  = cfg.MODEL.pretrained,
        **cfg.MODEL.get('params',{})
    )
    
    # optimizer
    optimizer = create_optimizer(
        opt_name   = cfg.OPTIMIZER.name, 
        model      = model.prompt if cfg.MODEL.name == 'VPTAL' else model, 
        lr         = cfg.OPTIMIZER.lr, 
        opt_params = cfg.OPTIMIZER.get('params',{})
    )

    # scheduler
    scheduler = create_scheduler(
        sched_name    = cfg.SCHEDULER.name, 
        optimizer     = optimizer, 
        epochs        = cfg.TRAIN.epochs, 
        params        = cfg.SCHEDULER.params,
        warmup_params = cfg.SCHEDULER.get('warmup_params', {})
    )

    # criterion 
    criterion = create_criterion(name=cfg.LOSS.name, params=cfg.LOSS.get('params', {}))
    
    # prepraring accelerator
    model, query_model, optimizer, trainloader, validloader, testloader, scheduler = accelerator.prepare(
        model, query_model, optimizer, trainloader, validloader, testloader, scheduler
    )

    # fitting model
    fit(
        model        = model, 
        query_model  = query_model,
        trainloader  = trainloader, 
        testloader   = validloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        accelerator  = accelerator,
        epochs       = cfg.TRAIN.epochs, 
        use_wandb    = cfg.TRAIN.wandb.use,
        log_interval = cfg.TRAIN.log_interval,
        seed         = cfg.DEFAULT.seed
    )
    
    # save model
    if cfg.MODEL.name == 'VPTAL':
        torch.save(model.prompt.state_dict(), os.path.join(savedir, f"prompt_seed{cfg.DEFAULT.seed}.pt"))
    else:
        torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{cfg.DEFAULT.seed}.pt"))

    # ====================
    # validation results
    # ====================
    
    eval_results = test(
        model            = model, 
        query_model      = query_model,
        dataloader       = validloader, 
        criterion        = criterion, 
        log_interval     = cfg.TRAIN.log_interval,
        return_per_class = True
    )

    # save results per class
    json.dump(
        obj    = eval_results['per_class'], 
        fp     = open(os.path.join(savedir, f"results-seed{cfg.DEFAULT.seed}_valid-per_class.json"), 'w'), 
        cls    = MyEncoder,
        indent = '\t'
    )
    del eval_results['per_class']

    # save results
    json.dump(eval_results, open(os.path.join(savedir, f'results-seed{cfg.DEFAULT.seed}_valid.json'), 'w'), indent='\t')
    

    # ====================
    # test results
    # ====================
    
    test_results = test(
        model            = model, 
        query_model      = query_model,
        dataloader       = testloader, 
        criterion        = criterion, 
        log_interval     = cfg.TRAIN.log_interval,
        return_per_class = True
    )

    # save results per class
    json.dump(
        obj    = test_results['per_class'], 
        fp     = open(os.path.join(savedir, f"results-seed{cfg.DEFAULT.seed}_test-per_class.json"), 'w'), 
        cls    = MyEncoder,
        indent = '\t'
    )
    del test_results['per_class']

    # save results
    json.dump(test_results, open(os.path.join(savedir, f'results-seed{cfg.DEFAULT.seed}_test.json'), 'w'), indent='\t')
    

def al_run(cfg: dict, trainset, validset, testset, savedir: str, accelerator: Accelerator):

    # set active learning arguments
    nb_round = (cfg.AL.n_end - cfg.AL.n_start)/cfg.AL.n_query
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # logging
    _logger.info('[total samples] {}, [initial samples] {} [query samples] {} [end samples] {} [total round] {}'.format(
        len(trainset), cfg.AL.n_start, cfg.AL.n_query, cfg.AL.n_end, nb_round))
    
    # inital sampling labeling
    is_labeled = create_is_labeled(
        method   = cfg.AL.init.method,
        trainset = trainset,
        size     = cfg.AL.n_start,
        seed     = cfg.DEFAULT.seed,
        **cfg.AL.init.get('params', {})
    )
    
    # load model
    query_model, model = create_model(
        modelname   = cfg.MODEL.name,
        num_classes = cfg.DATASET.num_classes, 
        pretrained  = cfg.MODEL.pretrained, 
        **cfg.MODEL.get('params',{})
    )
    
    # select strategy    
    trainloader_type = 'epoch'
    if cfg.TRAIN.get('params'):
        if cfg.TRAIN.params.get('steps_per_epoch'):
            trainloader_type = 'step'
            
    strategy = create_query_strategy(
        strategy_name    = cfg.AL.strategy, 
        model            = model,
        dataset          = trainset, 
        test_transform   = testset.transform,
        sampler_name     = cfg.DATASET.sampler_name,
        is_labeled       = is_labeled, 
        n_query          = cfg.AL.n_query, 
        n_subset         = cfg.AL.n_subset,
        batch_size       = cfg.DATASET.batch_size, 
        num_workers      = cfg.DATASET.num_workers,
        trainloader_type = trainloader_type,
        tta_agg          = cfg.AL.get('tta_agg', None),
        tta_params       = cfg.AL.get('tta_params', None),
        interval_type    = cfg.AL.get('interval_type', 'top'),
        resampler_params = cfg.AL.get('resampler_params', None),
        **cfg.AL.get('params', {})
    )
    
    # define train dataloader
    trainloader = strategy.get_trainloader()
    
    # define log dataframe
    log_df_valid = pd.DataFrame(
        columns=['round', 'auroc', 'f1', 'recall', 'precision', 'bcr', 'acc', 'loss']
    )
    log_df_test = pd.DataFrame(
        columns=['round', 'auroc', 'f1', 'recall', 'precision', 'bcr', 'acc', 'loss']
    )
    
    # query log dataframe
    query_log_df = pd.DataFrame({'idx': range(len(is_labeled))})
    query_log_df['query_round'] = None
    query_log_df.loc[is_labeled, 'query_round'] = 'round0'
    
    # number of labeled set log dataframe
    nb_labeled_df = pd.DataFrame({'round': range(nb_round+1), 'nb_labeled': [0]*(nb_round+1)})
    nb_labeled_df.loc[nb_labeled_df['round']==0, 'nb_labeled'] = is_labeled.sum()
    
    # define dict to save confusion matrix and metrics per class
    metrics_log_eval = {}
    metrics_log_test = {}
    
    # run
    for r in range(nb_round+1):
        
        if r != 0:    
            # query sampling    
            query_params = {}
            if cfg.AL.strategy == 'PromptEnsemble':
                query_params = {'r': r, 'seed': cfg.DEFAULT.seed, 'savedir': savedir}
                
            query_idx = strategy.query(model, **query_params)
            
            # update query
            strategy.update(query_idx)
            
            # query_idx resampling
            if strategy.use_resampler:
                strategy.resampler(model)
            
            # get trainloader
            del trainloader
            trainloader = strategy.get_trainloader()
            
            # save query index
            query_log_df.loc[query_idx, 'query_round'] = f'round{r}'
            query_log_df.to_csv(os.path.join(savedir, 'query_log.csv'), index=False)
            
            # save nb_labeled
            nb_labeled_df.loc[nb_labeled_df['round']==r, 'nb_labeled'] = strategy.is_labeled.sum()
            nb_labeled_df.to_csv(os.path.join(savedir, 'nb_labeled.csv'), index=False)
            
            # clean memory
            del optimizer, scheduler, validloader, testloader
            if not cfg.AL.continual:
                del model
                
            accelerator.free_memory()
            
        # logging
        _logger.info('[Round {}/{}] training samples: {}'.format(r, nb_round, sum(is_labeled)))
        
        # build Model
        if not cfg.AL.get('continual', False) or r == 0:
            model = strategy.init_model()
        
        # optimizer
        optimizer = create_optimizer(
            opt_name   = cfg.OPTIMIZER.name, 
            model      = model, 
            lr         = cfg.OPTIMIZER.lr, 
            opt_params = cfg.OPTIMIZER.get('params',{})
        )

        # scheduler
        scheduler = create_scheduler(
            sched_name    = cfg.SCHEDULER.name, 
            optimizer     = optimizer, 
            epochs        = cfg.TRAIN.epochs, 
            params        = cfg.SCHEDULER.params,
            warmup_params = cfg.SCHEDULER.get('warmup_params', {})
        )

        # define test dataloader
        validloader = DataLoader(
            dataset     = validset,
            batch_size  = cfg.DATASET.test_batch_size,
            shuffle     = False,
            num_workers = cfg.DATASET.num_workers
        )
        
        # define test dataloader
        testloader = DataLoader(
            dataset     = testset,
            batch_size  = cfg.DATASET.test_batch_size,
            shuffle     = False,
            num_workers = cfg.DATASET.num_workers
        )
        
        # prepraring accelerator
        model, query_model, optimizer, trainloader, validloader, testloader, scheduler = accelerator.prepare(
            model, query_model, optimizer, trainloader, validloader, testloader, scheduler
        )
        
        # initialize wandb
        if cfg.TRAIN.wandb.use:
            wandb.init(name=f'{cfg.DEFAULT.exp_name}_round{r}', project=cfg.TRAIN.wandb.project_name, entity=cfg.TRAIN.wandb.entity, config=OmegaConf.to_container(cfg))

        # fitting model
        fit(
            model        = model, 
            query_model  = query_model, 
            trainloader  = trainloader, 
            testloader   = None, 
            criterion    = strategy.loss_fn, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            epochs       = cfg.TRAIN.epochs, 
            use_wandb    = cfg.TRAIN.wandb.use,
            log_interval = cfg.TRAIN.log_interval,
            seed         = cfg.DEFAULT.seed,
            **cfg.TRAIN.get('params', {})
        )
        
        # save model
        if cfg.MODEL.name == 'VPTAL':
            torch.save(model.prompt.state_dict(), os.path.join(savedir, f"prompt_seed{cfg.DEFAULT.seed}-round{r}.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{cfg.DEFAULT.seed}-round{r}.pt"))

        # aggretate previous weights
        if cfg.AL.strategy == 'PromptEnsemble' and cfg.TRAIN.get('ensemble', False):
            prompt_weights = PromptEnsemble.weights_average(r=r+1, seed=cfg.DEFAULT.seed, savedir=savedir, weights=strategy.weights)
            model.prompt.load_state_dict(prompt_weights)

        # ====================
        # validation results
        # ====================
        
        if validset != testset:
            eval_results = test(
                model            = model, 
                query_model      = query_model,
                dataloader       = validloader, 
                criterion        = strategy.loss_fn, 
                log_interval     = cfg.TRAIN.log_interval,
                return_per_class = True,
                name             = 'VALID'
            )

            # save results per class
            metrics_log_eval.update({
                f'round{r}': eval_results['per_class']
            })
            json.dump(
                obj    = metrics_log_eval, 
                fp     = open(os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_valid-per_class.json"), 'w'), 
                cls    = MyEncoder,
                indent = '\t'
            )
            
            del eval_results['per_class']
            
            # save results 
            log_metrics = {'round':r}
            log_metrics.update([(k, v) for k, v in eval_results.items()])
            log_df_valid = pd.concat([log_df_valid, pd.DataFrame(log_metrics, index=[len(log_df_valid)])], axis=0)
            
            log_df_valid.round(4).to_csv(
                os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_valid.csv"),
                index=False
            )   

        # ====================
        # test results
        # ====================
        
        test_results = test(
            model            = model, 
            query_model      = query_model,
            dataloader       = testloader, 
            criterion        = strategy.loss_fn, 
            log_interval     = cfg.TRAIN.log_interval,
            return_per_class = True
        )

        # save results per class
        metrics_log_test.update({
            f'round{r}': test_results['per_class']
        })
        json.dump(
            obj    = metrics_log_test, 
            fp     = open(os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_test-per_class.json"), 'w'), 
            cls    = MyEncoder,
            indent = '\t'
        )
        
        del test_results['per_class']
        
        # save results 
        log_metrics = {'round':r}
        log_metrics.update([(k, v) for k, v in test_results.items()])
        log_df_test = pd.concat([log_df_test, pd.DataFrame(log_metrics, index=[len(log_df_test)])], axis=0)
        
        log_df_test.round(4).to_csv(
            os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_test.csv"),
            index=False
        )   
        
        
        _logger.info('append result [shape: {}]'.format(log_df_test.shape))
        
        wandb.finish()
        
        
        
def openset_al_run(cfg: dict, trainset, validset, testset, savedir: str, accelerator: Accelerator):

    # set active learning arguments
    nb_round = (cfg.AL.n_end - cfg.AL.n_start)/cfg.AL.n_query
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # logging
    _logger.info('[total samples] {}, [initial samples] {} [query samples] {} [end samples] {} [total round] {} [OOD ratio] {}'.format(
        len(trainset), cfg.AL.n_start, cfg.AL.n_query, cfg.AL.n_end, nb_round, cfg.AL.ood_ratio))
    
    # create ID and OOD targets
    trainset, id_targets = create_id_ood_targets(
        dataset     = trainset,
        nb_id_class = cfg.AL.nb_id_class,
        seed        = cfg.DEFAULT.seed
    )
    validset, id_targets_check = create_id_ood_targets(
        dataset     = validset,
        nb_id_class = cfg.AL.nb_id_class,
        seed        = cfg.DEFAULT.seed
    )
    assert sum(id_targets == id_targets_check) == cfg.AL.nb_id_class, "ID targets are not matched"
    testset, id_targets_check = create_id_ood_targets(
        dataset     = testset,
        nb_id_class = cfg.AL.nb_id_class,
        seed        = cfg.DEFAULT.seed
    )
    assert sum(id_targets == id_targets_check) == cfg.AL.nb_id_class, "ID targets are not matched"
    
    # save selected ID targets
    json.dump(
        obj    = {'target_ids': list(map(int, id_targets))},
        fp     = open(os.path.join(savedir, 'target_ids.json'), 'w'), 
        indent = '\t'
    )
    
    # inital sampling labeling
    is_labeled, is_unlabeled, is_ood = create_is_labeled_unlabeled(
        trainset   = trainset,
        id_targets = id_targets,
        size       = cfg.AL.n_start,
        ood_ratio  = cfg.AL.ood_ratio,
        seed       = cfg.DEFAULT.seed
    )
    
    # load model
    _, model = create_model(
        modelname   = cfg.MODEL.name,
        num_classes = cfg.DATASET.num_classes, 
        pretrained  = cfg.MODEL.pretrained, 
        **cfg.MODEL.get('params',{})
    )

    # select strategy    
    openset_params = {
        'is_openset'   : True,
        'is_unlabeled' : is_unlabeled,
        'is_ood'       : is_ood,
        'id_classes'   : trainset.classes[id_targets],
        'savedir'      : savedir,
        'seed'         : cfg.DEFAULT.seed
    }
    openset_params.update(cfg.AL.get('openset_params', {}))
    openset_params.update(cfg.AL.get('params', {}))
    
    trainloader_type = 'epoch'
    if cfg.TRAIN.get('params'):
        if cfg.TRAIN.params.get('steps_per_epoch'):
            trainloader_type = 'step'
    
    strategy = create_query_strategy(
        strategy_name    = cfg.AL.strategy, 
        model            = model,
        dataset          = trainset, 
        test_transform   = testset.transform,
        sampler_name     = cfg.DATASET.sampler_name,
        is_labeled       = is_labeled, 
        n_query          = cfg.AL.n_query, 
        n_subset         = cfg.AL.n_subset,
        batch_size       = cfg.DATASET.batch_size, 
        num_workers      = cfg.DATASET.num_workers,
        trainloader_type = trainloader_type,
        **openset_params
    )
    
    # define train dataloader
    trainloader = strategy.get_trainloader()
    
    # define log dataframe
    log_df_valid = pd.DataFrame(
        columns=['round', 'auroc', 'f1', 'recall', 'precision', 'bcr', 'acc', 'loss']
    )
    log_df_test = pd.DataFrame(
        columns=['round', 'auroc', 'f1', 'recall', 'precision', 'bcr', 'acc', 'loss']
    )
    
    # query log dataframe
    query_log_df = pd.DataFrame({'idx': range(len(is_labeled)), 'is_unlabel': np.zeros(len(is_labeled), dtype=bool)})
    query_log_df.loc[is_unlabeled, 'is_unlabel'] = True
    query_log_df['query_round'] = None
    query_log_df['ID_query_round'] = None
    query_log_df.loc[np.r_[np.where(is_labeled==True)[0], np.where(is_ood==True)[0]], 'query_round'] = 'round0'
    query_log_df.loc[is_labeled, 'ID_query_round'] = 'round0'
    
    # number of labeled set log dataframe
    nb_labeled_df = pd.DataFrame({'round': range(nb_round+1), 'nb_labeled': [0]*(nb_round+1)})
    nb_labeled_df.loc[nb_labeled_df['round']==0, 'nb_labeled'] = is_labeled.sum()
    
    # define dict to save confusion matrix and metrics per class
    metrics_log_eval = {}
    metrics_log_test = {}
    
    # run
    for r in range(nb_round+1):
        
        if r != 0:    
            # query sampling    
            query_idx = strategy.query(model)
            
            # update query
            id_query_idx = strategy.update(query_idx=query_idx)
            
            # query_idx resampling
            if strategy.use_resampler:
                strategy.resampler(model)
            
            # get trainloader
            del trainloader
            trainloader = strategy.get_trainloader()
            
            # save query index
            query_log_df.loc[query_idx, 'query_round'] = f'round{r}'
            query_log_df.loc[id_query_idx, 'ID_query_round'] = f'round{r}'
            query_log_df.to_csv(os.path.join(savedir, 'query_log.csv'), index=False)
            
            # save nb_labeled
            nb_labeled_df.loc[nb_labeled_df['round']==r, 'nb_labeled'] = strategy.is_labeled.sum()
            nb_labeled_df.to_csv(os.path.join(savedir, 'nb_labeled.csv'), index=False)
            
            # clean memory
            del optimizer, scheduler, validloader, testloader
            if not cfg.AL.continual:
                del model
                
            accelerator.free_memory()
            
        # logging
        _logger.info('[Round {}/{}] training samples: {}'.format(r, nb_round, sum(is_labeled)))
        
        # build Model          
        if not cfg.AL.get('continual', False) or r == 0:
            model = strategy.init_model()
        
        # optimizer
        optimizer = create_optimizer(
            opt_name   = cfg.OPTIMIZER.name, 
            model      = model, 
            lr         = cfg.OPTIMIZER.lr, 
            opt_params = cfg.OPTIMIZER.get('params',{})
        )

        # scheduler
        scheduler = create_scheduler(
            sched_name    = cfg.SCHEDULER.name, 
            optimizer     = optimizer, 
            epochs        = cfg.TRAIN.epochs, 
            params        = cfg.SCHEDULER.params,
            warmup_params = cfg.SCHEDULER.get('warmup_params', {})
        )


        # define test dataloader
        validloader = create_id_testloader(
            dataset     = validset,
            id_targets  = id_targets,
            batch_size  = cfg.DATASET.test_batch_size,
            num_workers = cfg.DATASET.num_workers    
        )
        testloader = create_id_testloader(
            dataset     = testset,
            id_targets  = id_targets,
            batch_size  = cfg.DATASET.test_batch_size,
            num_workers = cfg.DATASET.num_workers    
        )
        
        # prepraring accelerator
        model, optimizer, trainloader, validloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, validloader, testloader, scheduler
        )
        
        # initialize wandb
        if cfg.TRAIN.wandb.use:
            wandb.init(name=f'{cfg.DEFAULT.exp_name}_round{r}', project=cfg.TRAIN.wandb.project_name, entity=cfg.TRAIN.wandb.entity, config=OmegaConf.to_container(cfg))

        # fitting model
        fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = None, 
            criterion    = strategy.loss_fn, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            epochs       = cfg.TRAIN.epochs, 
            use_wandb    = cfg.TRAIN.wandb.use,
            log_interval = cfg.TRAIN.log_interval,
            seed         = cfg.DEFAULT.seed,
            **cfg.TRAIN.get('params', {})
        )
        
        # save model
        torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{cfg.DEFAULT.seed}-round{r}.pt"))

        # ====================
        # validation results
        # ====================
        
        if validset != testset:
            eval_results = test(
                model            = model, 
                dataloader       = validloader, 
                criterion        = strategy.loss_fn, 
                log_interval     = cfg.TRAIN.log_interval,
                return_per_class = True,
                name             = 'VALID'
            )

            # save results per class
            metrics_log_eval.update({
                f'round{r}': eval_results['per_class']
            })
            json.dump(
                obj    = metrics_log_eval, 
                fp     = open(os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_valid-per_class.json"), 'w'), 
                cls    = MyEncoder,
                indent = '\t'
            )
            
            del eval_results['per_class']
            
            # save results 
            log_metrics = {'round':r}
            log_metrics.update([(k, v) for k, v in eval_results.items()])
            log_df_valid = pd.concat([log_df_valid, pd.DataFrame(log_metrics, index=[len(log_df_valid)])], axis=0)
            
            log_df_valid.round(4).to_csv(
                os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_valid.csv"),
                index=False
            )   

        # ====================
        # test results
        # ====================
        
        test_results = test(
            model            = model, 
            dataloader       = testloader, 
            criterion        = strategy.loss_fn, 
            log_interval     = cfg.TRAIN.log_interval,
            return_per_class = True
        )

        # save results per class
        metrics_log_test.update({
            f'round{r}': test_results['per_class']
        })
        json.dump(
            obj    = metrics_log_test, 
            fp     = open(os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_test-per_class.json"), 'w'), 
            cls    = MyEncoder,
            indent = '\t'
        )
        
        del test_results['per_class']
        
        # save results 
        log_metrics = {'round':r}
        log_metrics.update([(k, v) for k, v in test_results.items()])
        log_df_test = pd.concat([log_df_test, pd.DataFrame(log_metrics, index=[len(log_df_test)])], axis=0)
        
        log_df_test.round(4).to_csv(
            os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_test.csv"),
            index=False
        )   
        
        _logger.info('append result [shape: {}]'.format(log_df_test.shape))
        
        wandb.finish()