import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict

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
    # check type
    if type(outputs) == torch.Tensor:
        outputs = outputs
    elif type(outputs) == dict:
        outputs = outputs['logits']
        
    # calculate accuracy
    preds = outputs.argmax(dim=1) 
    correct = targets.eq(preds).sum().item()
    
    if return_correct:
        return correct
    else:
        return correct/targets.size(0)


def train(model, dataloader, criterion, optimizer, accelerator, log_interval: int) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
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
    
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])
        
        
def test(model, dataloader, criterion, log_interval: int) -> dict:
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            # predict
            outputs = model(inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            correct += accuracy(outputs, targets, return_correct=False)
            total += targets.size(0)
            
            if (idx+1) % log_interval == 0: 
                _logger.info('TEST [{0:d}/{1:d}]: Loss: {2:.3f} | Acc: {3:.3f}% [{4:d}/{5:d}]'.format(idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
                
    return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader))])
                
def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, savedir: str = None
) -> None:

    step = 0
    best_acc = 0
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
            log_interval = log_interval
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
        if savedir and (best_acc < eval_metrics['acc']):
            best_acc = eval_metrics['acc']
            state = {'best_step':step}
            state.update(eval_metrics)
            json.dump(state, open(os.path.join(savedir, 'best_results.json'), 'w'), indent='\t')
            torch.save(model.state_dict(), os.path.join(savedir, 'model.pt'))
    
    return eval_metrics
