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

from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, \
                            balanced_accuracy_score, classification_report, confusion_matrix

from query_strategies import create_query_strategy, create_is_labeled, \
                             create_id_ood_targets, create_is_labeled_unlabeled, \
                             create_id_testloader, torch_seed, \
                             create_scheduler, create_optimizer
from query_strategies.utils import TrainIterableDataset
from models import create_model
from query_strategies.utils import NoIndent, MyEncoder
from omegaconf import OmegaConf

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

def create_criterion(name, params):
    if name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss(**params)

def calc_metrics(y_true: list, y_score: np.ndarray, y_pred: list, return_per_class: bool = False) -> dict:
    # softmax
    y_score = torch.nn.functional.softmax(torch.FloatTensor(y_score), dim=1)
    
    # metrics
    if y_score.shape[1] == 2: # binary
        auroc = roc_auc_score(y_true, y_score[:,1])
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        bcr = balanced_accuracy_score(y_true, y_pred)
    else: # multi-class
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
        if y_score.shape[1] == 2: # binary
            f1_per_class = f1_score(y_true, y_pred)
            recall_per_class = recall_score(y_true, y_pred)
            precision_per_class = precision_score(y_true, y_pred)
        else: # multi-class
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

def load_resume(ckp_path: str, ckp_round: int, strategy, seed: int, is_openset: bool = False):
    total, init, query = os.path.basename(os.path.dirname(ckp_path)).split('-')
    n_total = int(total.split('_')[1])
    n_init = int(init.split('_')[1])
    n_query = int(query.split('_')[1])
    
    # get round
    nb_round = (n_total - n_init)/n_query
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # load previous results
    model = strategy.init_model()
    model.load_state_dict(torch.load(os.path.join(ckp_path, f'model_seed{seed}-round{ckp_round}.pt')))
    
    # history query
    query_log_df = pd.read_csv(os.path.join(ckp_path, 'query_log.csv'))
    round_list = [f'round{r}' for r in range(ckp_round+1)]
    query_log_df.loc[~query_log_df['query_round'].isin(round_list), 'query_round'] = np.nan
    
    nb_labeled_df = pd.read_csv(os.path.join(ckp_path, 'nb_labeled.csv'))
    nb_labeled_df = nb_labeled_df.iloc[:ckp_round+1]
    
    # history metrics
    log_df_valid = pd.read_csv(os.path.join(ckp_path, f'round{nb_round}-seed{seed}_valid.csv')).iloc[:ckp_round+1]
    log_df_test = pd.read_csv(os.path.join(ckp_path, f'round{nb_round}-seed{seed}_test.csv')).iloc[:ckp_round+1]
    
    metrics_log_eval = dict([
        (k, v) for k, v in json.load(open(os.path.join(ckp_path, f'round{nb_round}-seed{seed}_valid-per_class.json'), 'r')).items() 
        if int(k.strip('round')) <= ckp_round
    ])
    metrics_log_test = dict([
        (k, v) for k, v in json.load(open(os.path.join(ckp_path, f'round{nb_round}-seed{seed}_test-per_class.json'), 'r')).items() 
        if int(k.strip('round')) <= ckp_round
    ])
    
    # reset index 
    if is_openset:
        query_log_df.loc[~query_log_df['ID_query_round'].isin(round_list), 'ID_query_round'] = np.nan
        
        query_idx = query_log_df[~query_log_df['ID_query_round'].isna()]['idx'].values
        all_query_idx = query_log_df[~query_log_df['query_round'].isna()]['idx'].values
        unlabeled_idx = query_log_df[query_log_df['query_round'].isna()]['idx'].values
        ood_query_idx = np.array(list(set(all_query_idx) - set(query_idx)))
        
        # init
        strategy.is_ood[:] = False
        strategy.is_unlabeled[:] = False
        
        # reset
        strategy.is_ood[ood_query_idx] = True
        strategy.is_unlabeled[unlabeled_idx] = True
        
        print("[RESUME] # Labeled in: {}, ood: {}, Unlabeled: {}".format(len(query_idx), len(ood_query_idx), len(unlabeled_idx)))
    else:
        query_idx = query_log_df[~query_log_df['query_round'].isna()]['idx'].values
        unlabeled_idx = query_log_df[query_log_df['query_round'].isna()]['idx'].values
        
        print("[RESUME] # Labeled: {}, Unlabeled: {}".format(len(query_idx), len(unlabeled_idx)))
    
    # init
    strategy.is_labeled[:] = False
    
    # reset
    strategy.is_labeled[query_idx] = True
    
    start_round = ckp_round + 1
    
    return_output = {
        'model'       : model,
        'strategy'    : strategy,
        'start_round' : start_round,
        'history'     : {
            'query_log'  : query_log_df,
            'nb_labeled' : nb_labeled_df,
            'log' : {
                'valid' : log_df_valid,
                'test'  : log_df_test
            },
            'metrics' : {
                'valid' : metrics_log_eval,
                'test'  : metrics_log_test
            }
        }
    }
    
    return return_output


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
    print(classification_report(y_true=targets, y_pred=preds, digits=4, zero_division=0.0))
    
    return metrics, metrics_log



def train(model, dataloader, criterion, optimizer, accelerator: Accelerator, log_interval: int, **train_params) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    total_preds = []
    total_score = []
    total_targets = []
    
    end = time.time()
    
    model.train()
    if isinstance(optimizer, dict):
        for k in optimizer.keys():
            optimizer[k].zero_grad()
    else:
        optimizer.zero_grad()
    
    steps_per_epoch = train_params.get('steps_per_epoch') if train_params.get('steps_per_epoch') else len(dataloader)
    
    step = 0
    for idx, (inputs, targets) in enumerate(dataloader):
        with accelerator.accumulate(model):
            data_time_m.update(time.time() - end)
            
            # predict
            if getattr(model, 'LPM', False): # learning loss
                outputs = {}
                outputs['logits'] = model.backbone(inputs)

                # detach LPM for learning loss
                if train_params.get('is_detach_lpm', False):
                    for k, v in model.layer_outputs.items():
                        model.layer_outputs[k] = v.detach()
                
                outputs['loss_pred'] = model.LPM(model.layer_outputs)

            else:
                outputs = model(inputs)        

            # calc loss
            if criterion.__class__.__name__ == 'CrossEntropyLoss' and isinstance(outputs, dict):
                outputs = outputs['logits']
                
            loss = criterion(outputs, targets)    
            accelerator.backward(loss)

            # loss update
            if isinstance(optimizer, dict):
                for k in optimizer.keys():
                    optimizer[k].step()
                    optimizer[k].zero_grad()
            else:
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
                    if isinstance(optimizer, dict):
                        lr_current = optimizer['backbone'].param_groups[0]['lr']
                    else:
                        lr_current = optimizer.param_groups[0]['lr']
                    print('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                                'Acc: {acc.avg:.3%} '
                                'LR: {lr:.3e} '
                                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                (step+1)//accelerator.gradient_accumulation_steps, 
                                steps_per_epoch//accelerator.gradient_accumulation_steps, 
                                loss       = losses_m, 
                                acc        = acc_m, 
                                lr         = lr_current,
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
    print(metrics_log)
        
    return metrics

        
def test(model, dataloader, criterion, log_interval: int, name: str = 'TEST', return_per_class: bool = False) -> dict:
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
                print('{0:s} [{1:d}/{2:d}]: Loss: {3:.3f} | Acc: {4:.3f}% [{5:d}/{6:d}]'.format(name, idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
    
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
    print(metrics_log)
    
    return metrics
            
                
def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, accelerator: Accelerator,
    epochs: int, use_wandb: bool, log_interval: int, seed: int = 0, **train_params
) -> None:

    step = 0
    
    torch_seed(seed)    
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch+1}/{epochs}')
        
        # for learning loss
        if 'detach_epoch_ratio' in train_params.keys():
            is_detach_lpm = True if epoch > int(epochs * train_params['detach_epoch_ratio']) else False
            train_params['is_detach_lpm'] = is_detach_lpm
        
        train_metrics = train(
            model        = model, 
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
                dataloader   = testloader, 
                criterion    = criterion, 
                log_interval = log_interval,
                name         = 'VALID'
            )

        # wandb
        if use_wandb:
            metrics = OrderedDict(lr=optimizer['backbone'].param_groups[0]['lr'])
            metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
            
            if testloader != None:
                metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
                
            wandb.log(metrics, step=step)
        
        step += 1
        
        # update scheduler  
        if scheduler:
            if isinstance(scheduler, dict):
                for k in scheduler.keys():
                    scheduler[k].step()
            else:
                scheduler.step()

def full_run(cfg: dict, trainset, validset, testset, savedir: str):
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )
    
    # set device
    print('Device: {}'.format(accelerator.device))
    
    # id_ratio
    if cfg.DATASET.get('id_ratio', False):
        # create ID and OOD targets
        trainset, id_targets = create_id_ood_targets(
            dataset     = trainset,
            nb_id_class = cfg.DATASET.nb_id_class,
            seed        = cfg.DEFAULT.seed
        )
        validset, id_targets_check = create_id_ood_targets(
            dataset     = validset,
            nb_id_class = cfg.DATASET.nb_id_class,
            seed        = cfg.DEFAULT.seed
        )
        assert sum(id_targets == id_targets_check) == cfg.DATASET.nb_id_class, "ID targets are not matched"
        testset, id_targets_check = create_id_ood_targets(
            dataset     = testset,
            nb_id_class = cfg.DATASET.nb_id_class,
            seed        = cfg.DEFAULT.seed
        )
        assert sum(id_targets == id_targets_check) == cfg.DATASET.nb_id_class, "ID targets are not matched"
    
        # define dataloader
        train_idx = [i for i in range(len(trainset.targets)) if trainset.targets[i] < len(id_targets)]
        if cfg.TRAIN.get('params', {}).get('steps_per_epoch', False):
            trainset = TrainIterableDataset(
                dataset    = trainset,
                sample_idx = train_idx
            )
            
            trainloader = DataLoader(
                dataset     = trainset,
                batch_size  = cfg.DATASET.batch_size,
                num_workers = cfg.DATASET.num_workers
            )
        else:
            trainloader = DataLoader(
                dataset     = trainset,
                batch_size  = cfg.DATASET.batch_size,
                sampler     = SubsetRandomSampler(indices=train_idx),
                num_workers = cfg.DATASET.num_workers
            )
        
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

    else:
        # define dataloader
        if cfg.TRAIN.get('params', {}).get('steps_per_epoch', False):
            trainloader = DataLoader(
                dataset     = TrainIterableDataset(dataset=trainset),
                batch_size  = cfg.DATASET.batch_size,
                num_workers = cfg.DATASET.num_workers
            )
        else:
            trainloader = DataLoader(
                dataset     = trainset,
                batch_size  = cfg.DATASET.batch_size,
                shuffle     = True,
                num_workers = cfg.DATASET.num_workers
            )
        
        validloader = DataLoader(
            dataset     = validset,
            batch_size  = cfg.DATASET.test_batch_size,
            shuffle     = False,
            num_workers = cfg.DATASET.num_workers
        )
        
        testloader = DataLoader(
            dataset     = testset,
            batch_size  = cfg.DATASET.test_batch_size,
            shuffle     = False,
            num_workers = cfg.DATASET.num_workers
        )

    # load model
    model = create_model(
        modelname   = cfg.MODEL.name, 
        num_classes = cfg.DATASET.num_classes, 
        pretrained  = cfg.MODEL.pretrained,
        img_size    = cfg.DATASET.img_size,
        **cfg.MODEL.get('params',{})
    )
    
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
        params        = cfg.SCHEDULER.get('params', {}),
        warmup_params = cfg.SCHEDULER.get('warmup_params', {})
    )

    # criterion 
    criterion = create_criterion(name=cfg.LOSS.name, params=cfg.LOSS.get('params', {}))
    
    # prepraring accelerator
    model, trainloader, validloader, testloader = accelerator.prepare(
        model, trainloader, validloader, testloader
    )
    
    for k, opt in optimizer.items():
        optimizer[k] = accelerator.prepare(opt)
        
    for k, sched in scheduler.items():
        scheduler[k] = accelerator.prepare(sched)

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
        seed         = cfg.DEFAULT.seed,
        **cfg.TRAIN.get('params', {})
    )
    
    # save model
    torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{cfg.DEFAULT.seed}.pt"))

    # ====================
    # validation results
    # ====================
    
    eval_results = test(
        model            = model, 
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
    

def al_run(cfg: dict, trainset, validset, testset, savedir: str):
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )
    
    # set device
    print('Device: {}'.format(accelerator.device))
    
    # set active learning arguments
    nb_round = (cfg.AL.n_end - cfg.AL.n_start)/cfg.AL.n_query
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # logging
    print('[total samples] {}, [initial samples] {} [query samples] {} [end samples] {} [total round] {}'.format(
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
    model = create_model(
        modelname   = cfg.MODEL.name,
        num_classes = cfg.DATASET.num_classes, 
        pretrained  = cfg.MODEL.pretrained, 
        img_size    = cfg.DATASET.img_size,
        **cfg.MODEL.get('params',{})
    )
    
    # select strategy                
    strategy = create_query_strategy(
        strategy_name    = cfg.AL.strategy, 
        model            = model,
        dataset          = trainset, 
        transform        = testset.transform,
        sampler_name     = cfg.DATASET.sampler_name,
        is_labeled       = is_labeled, 
        n_query          = cfg.AL.n_query, 
        n_subset         = cfg.AL.n_subset,
        batch_size       = cfg.DATASET.batch_size, 
        num_workers      = cfg.DATASET.num_workers,
        steps_per_epoch  = cfg.TRAIN.params.get('steps_per_epoch', 0),
        interval_type    = cfg.AL.get('interval_type', 'top'),
        **cfg.AL.get('params', {})
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
    
    # resume
    if cfg.TRAIN.get('resume', False).get('use', False):
        is_resume = True
        
        resume_info = load_resume(
            ckp_path   = cfg.TRAIN.resume.ckp_path,
            ckp_round  = cfg.TRAIN.resume.ckp_round,
            strategy   = strategy,
            seed       = cfg.DEFAULT.seed,
            is_openset = cfg.AL.get('ood_ratio', False)
        )
        
        model = resume_info['model']
        strategy = resume_info['strategy']
        start_round = resume_info['start_round']
        history = resume_info['history']
        
        query_log_df, nb_labeled_df = history['query_log'], history['nb_labeled']
        log_df_valid, log_df_test = history['log']['valid'], history['log']['test']
        metrics_log_eval, metrics_log_test = history['metrics']['valid'], history['metrics']['test']
        
        model = accelerator.prepare(model)
    else:
        is_resume = False
        start_round = 0 
        
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
        nb_labeled_df = pd.DataFrame({'round': [0], 'nb_labeled': [is_labeled.sum()]})
        
        # define dict to save confusion matrix and metrics per class
        metrics_log_eval = {}
        metrics_log_test = {}
        
    nb_round += start_round
    
    # run
    for r in range(start_round, nb_round+1):
        
        if r != 0:
            # query sampling    
            query_idx = strategy.query(model)
            
            # update query
            strategy.update(query_idx)
            
            # check resume
            if is_resume:
                is_resume = False
            
            # define accelerator
            accelerator = Accelerator(
                gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
                mixed_precision             = cfg.TRAIN.mixed_precision
            )
            if hasattr(strategy, 'accelerator'):
                strategy.accelerator = accelerator
            
            # save query index
            query_log_df.loc[query_idx, 'query_round'] = f'round{r}'
            query_log_df.to_csv(os.path.join(savedir, 'query_log.csv'), index=False)
            
            # save nb_labeled
            nb_labeled_r = {'round': [r], 'nb_labeled': [strategy.is_labeled.sum()]}
            nb_labeled_df = pd.concat([nb_labeled_df, pd.DataFrame(nb_labeled_r, index=[len(nb_labeled_df)])], axis=0)
            nb_labeled_df.to_csv(os.path.join(savedir, 'nb_labeled.csv'), index=False)
        
        # logging
        print('[Round {}/{}] training samples: {}'.format(r, nb_round, sum(strategy.is_labeled)))
        
        # build Model
        if not cfg.AL.get('continual', False) or r == 0:
            model = strategy.init_model()
            model = accelerator.prepare(model)
        
        # get trainloader
        trainloader = strategy.get_trainloader()
        trainloader, validloader, testloader = accelerator.prepare(trainloader, validloader, testloader)
        
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
            params        = cfg.SCHEDULER.get('params', {}),
            warmup_params = cfg.SCHEDULER.get('warmup_params', {})
        )
        for k, opt in optimizer.items():
            optimizer[k] = accelerator.prepare(opt)
            
        for k, sched in scheduler.items():
            scheduler[k] = accelerator.prepare(sched)
    
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
        
        
        print('append result [shape: {}]'.format(log_df_test.shape))
        
        wandb.finish()
        
        
        
def openset_al_run(cfg: dict, trainset, validset, testset, savedir: str):

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )
    
    # set device
    print('Device: {}'.format(accelerator.device))
    

    # set active learning arguments
    nb_round = (cfg.AL.n_end - cfg.AL.n_start)/cfg.AL.n_query
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # logging
    print('[total samples] {}, [initial samples] {} [query samples] {} [end samples] {} [total round] {} [OOD ratio] {}'.format(
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
    model = create_model(
        modelname   = cfg.MODEL.name,
        num_classes = cfg.DATASET.num_classes, 
        pretrained  = cfg.MODEL.pretrained, 
        img_size    = cfg.DATASET.img_size,
        **cfg.MODEL.get('params',{})
    )

    # select strategy    
    openset_params = {
        'is_openset'      : True,
        'is_unlabeled'    : is_unlabeled,
        'is_ood'          : is_ood,
        'id_classes'      : trainset.classes[id_targets],
        'savedir'         : savedir,
        'seed'            : cfg.DEFAULT.seed,
        'accelerator'     : accelerator
    }
    openset_params.update(cfg.AL.get('openset_params', {}))
    openset_params.update(cfg.AL.get('params', {}))
    
    strategy = create_query_strategy(
        strategy_name    = cfg.AL.strategy, 
        model            = model,
        dataset          = trainset, 
        transform        = testset.transform,
        sampler_name     = cfg.DATASET.sampler_name,
        is_labeled       = is_labeled, 
        n_query          = cfg.AL.n_query, 
        n_subset         = cfg.AL.n_subset,
        batch_size       = cfg.DATASET.batch_size, 
        num_workers      = cfg.DATASET.num_workers,
        steps_per_epoch  = cfg.TRAIN.params.get('steps_per_epoch', 0),
        **openset_params
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
    
    # resume
    if cfg.TRAIN.get('resume', False).get('use', False):
        is_resume = True
        
        resume_info = load_resume(
            ckp_path   = cfg.TRAIN.resume.ckp_path,
            ckp_round  = cfg.TRAIN.resume.ckp_round,
            strategy   = strategy,
            seed       = cfg.DEFAULT.seed,
            is_openset = cfg.AL.get('ood_ratio', False)
        )
        
        model = resume_info['model']
        strategy = resume_info['strategy']
        start_round = resume_info['start_round']
        history = resume_info['history']
        
        query_log_df, nb_labeled_df = history['query_log'], history['nb_labeled']
        log_df_valid, log_df_test = history['log']['valid'], history['log']['test']
        metrics_log_eval, metrics_log_test = history['metrics']['valid'], history['metrics']['test']
        
        model = accelerator.prepare(model)
        
    else:
        is_resume = False
        start_round = 0 
        
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
        nb_labeled_df = pd.DataFrame({'round': [0], 'nb_labeled': [is_labeled.sum()]})
        
        # define dict to save confusion matrix and metrics per class
        metrics_log_eval = {}
        metrics_log_test = {}
        
    nb_round += start_round
    
    # run
    for r in range(start_round, nb_round+1):
        
        if r != 0:    
            # query sampling    
            query_idx = strategy.query(model)
            
            # update query
            id_query_idx = strategy.update(query_idx=query_idx)
            
            # check resume
            if is_resume:
                is_resume = False
            
            # define accelerator
            accelerator = Accelerator(
                gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
                mixed_precision             = cfg.TRAIN.mixed_precision
            )
            if hasattr(strategy, 'accelerator'):
                strategy.accelerator = accelerator
            
            # save query index
            query_log_df.loc[query_idx, 'query_round'] = f'round{r}'
            query_log_df.loc[id_query_idx, 'ID_query_round'] = f'round{r}'
            query_log_df.to_csv(os.path.join(savedir, 'query_log.csv'), index=False)
            
            # save nb_labeled
            nb_labeled_r = {'round': [r], 'nb_labeled': [strategy.is_labeled.sum()]}
            nb_labeled_df = pd.concat([nb_labeled_df, pd.DataFrame(nb_labeled_r, index=[len(nb_labeled_df)])], axis=0)
            nb_labeled_df.to_csv(os.path.join(savedir, 'nb_labeled.csv'), index=False)
            
        # logging
        print('[Round {}/{}] training samples: {}'.format(r, nb_round, sum(strategy.is_labeled)))
        
        # build Model          
        if not cfg.AL.get('continual', False) or r == 0:
            model = strategy.init_model()
            model = accelerator.prepare(model)
        
        # get trainloader
        trainloader = strategy.get_trainloader()
        trainloader, validloader, testloader = accelerator.prepare(trainloader, validloader, testloader)
        
        # optimizer
        optimizer = create_optimizer(
            opt_name   = cfg.OPTIMIZER.name, 
            model      = model, 
            lr         = cfg.OPTIMIZER.lr, 
            opt_params = cfg.OPTIMIZER.get('params',{}),
            backbone   = True
        )

        # scheduler
        scheduler = create_scheduler(
            sched_name    = cfg.SCHEDULER.name, 
            optimizer     = optimizer, 
            epochs        = cfg.TRAIN.epochs, 
            params        = cfg.SCHEDULER.get('params', {}),
            warmup_params = cfg.SCHEDULER.get('warmup_params', {})
        )
        for k, opt in optimizer.items():
            optimizer[k] = accelerator.prepare(opt)
            
        for k, sched in scheduler.items():
            scheduler[k] = accelerator.prepare(sched)

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
        
        print('append result [shape: {}]'.format(log_df_test.shape))
        
        wandb.finish()