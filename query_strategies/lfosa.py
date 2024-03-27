import os
import json
import numpy as np
from glob import glob
from copy import deepcopy
from tqdm.auto import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from models import create_model
from .sampler import SubsetSequentialSampler
from .strategy import Strategy
from .utils import TrainIterableDataset, get_target_from_dataset, NoIndent, MyEncoder
from .optims import create_optimizer
from .scheds import create_scheduler

class LfOSA(Strategy):
    def __init__(
        self, 
        savedir: str,
        max_iter: int = 10, 
        tol: float = 1e-2, 
        reg_covar: float = 5e-4, 
        detector_params: dict = {},
        accelerator = None,
        **init_params
    ):
        
        super(LfOSA, self).__init__(**init_params)
        
        self.savedir = savedir
        self.accelerator = accelerator
        
        # GMM
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        
        self.gmm = GaussianMixture
        
        # detector
        self.detect = Detector(
            modelname       = detector_params['modelname'],
            num_classes     = self.num_id_class+1,
            epochs          = detector_params['epochs'],
            steps_per_epoch = detector_params.get('steps_per_epoch', 0),
            batch_size      = detector_params['batch_size'],
            num_workers     = detector_params['num_workers'],
            temperature     = detector_params['temperature'],
            lr              = detector_params['lr'],
            opt_name        = detector_params['opt_name'],
            sched_name      = detector_params['sched_name'],
            savedir         = self.savedir,
            opt_params      = detector_params['opt_params'],
            sched_params    = detector_params['sched_params'],
            warmup_params   = detector_params.get('warmup_params', {}),
            accelerator     = self.accelerator
        )
        
        
    def query(self, model, **kwargs) -> np.ndarray:
        '''
        1. train detecor with labeled and ood samples
        2. get maximum activation values(MAVs) for unlabeled samples
        3. train GMM using MAVs
        4. get probs of GMMs w/ argmax of gmm.means_
        5. get query
        
        '''
        unlabeled_idx = self.get_unlabeled_idx()
        
        # device
        device = next(model.parameters()).device
        
        # training detector
        detector = self.detect.init_detector(device=device)
        self.detect.fit_detector(
            detector   = detector, 
            dataset    = self.dataset,
            sample_idx = np.r_[np.where(self.is_labeled==True)[0], np.where(self.is_ood==True)[0]],
            device     = device
        )
        
        # get miximum activation values(MAVs) and predicted class for unlabeled samples
        mavs, preds = self.get_mav_and_prediction(
            detector      = detector,
            unlabeled_idx = unlabeled_idx,
            device        = device
        )
        
        # predict probs using GMMs trained on MAVs for each class
        gmm_probs = self.get_probs_from_gmm(mavs=mavs, preds=preds)
        self.check_ood_acc(
            id_pred_idx    = np.where(preds<self.num_id_class)[0],
            ulb_sample_idx = unlabeled_idx, 
            savedir        = self.savedir
        )
        
        # get query
        score_rank = gmm_probs.sort(descending=True)[1]
        select_idx = unlabeled_idx[score_rank[:self.n_query]]
        
        return select_idx
        
        
    def get_probs_from_gmm(self, mavs: np.ndarray, preds: np.ndarray):
        gmm_probs = np.zeros_like(mavs)
        
        unique_preds = torch.unique(preds)
        for c in tqdm(unique_preds, total=len(unique_preds), desc='Get probs from GMM of each class'):
            c_idx = torch.where(preds==c)[0]
            if len(c_idx) < 2:
                continue
            
            # the probs of predicted unknown class is zero
            if c == self.num_id_class:
                continue
            
            mavs_c = mavs[c_idx].unsqueeze(dim=1).numpy()        
            
            # GMM train and prediction
            gmm_c = self.gmm(n_components=2, max_iter=self.max_iter, tol=self.tol, reg_covar=self.reg_covar)
            gmm_c.fit(mavs_c)
            probs_c = gmm_c.predict_proba(mavs_c)
            probs_c = probs_c[:, gmm_c.means_.argmax()]
            
            # get probs for predicted class c
            gmm_probs[c_idx] = probs_c

        return torch.Tensor(gmm_probs)
        
    def get_mav_and_prediction(self, detector, unlabeled_idx, device: str):
        dataset = deepcopy(self.dataset)
        dataset.transform = self.test_transform
        
        # define sampler
        sampler = SubsetSequentialSampler(indices=unlabeled_idx)
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        # eval mode on
        detector.eval()
        
        # get MAVs and predictions
        all_mavs = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, total=len(dataloader), desc='Get MAVs and Prediction'):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = detector(inputs)
                mavs, preds = outputs.max(dim=1)
                
                all_mavs.append(mavs.cpu())
                all_preds.append(preds.cpu())
                
        all_mavs = torch.hstack(all_mavs)
        all_preds = torch.hstack(all_preds)
        
        return all_mavs, all_preds
        
    def check_ood_acc(self, id_pred_idx: np.ndarray, ulb_sample_idx: np.ndarray, savedir: str = None):
        
        # get unlabeled targets
        targets = get_target_from_dataset(self.dataset)
        ulb_targets = targets[ulb_sample_idx]
        
        # get ID and OOD index from unlabeled samples
        id_idx = [i for i, t in enumerate(ulb_targets) if t < self.num_id_class]
        ood_idx = [i for i, t in enumerate(ulb_targets) if t == self.num_id_class]
        nb_ulb = len(id_idx) + len(ood_idx)
        
        # set predictions
        y_pred = np.zeros(len(targets))
        y_pred[id_pred_idx] = 1 # ID: 1, OOD: 0
        y_pred = y_pred[ulb_sample_idx]
        
        # set targets
        y_true = np.array([1 if t < self.num_id_class else 0 for t in ulb_targets])
        
        # calc acc
        total_acc = (y_pred == y_true).sum() / nb_ulb
        id_acc = (y_pred[id_idx] == y_true[id_idx]).sum() / len(id_idx)
        ood_acc = (y_pred[ood_idx] == y_true[ood_idx]).sum() / len(ood_idx)
        
        # f1 score
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
    
        # confusion matrix
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    
        # results
        results = {
            'cm': [NoIndent(elem) for elem in cm.tolist()],
            'f1': f1,
            'acc': {
                'total' : total_acc, 
                'id'    : id_acc, 
                'ood'   : ood_acc, 
            },
            'num_samples': {
                'total' : nb_ulb, 
                'id'    : len(id_idx), 
                'ood'   : len(ood_idx)   
            }
        }
    
        if savedir:
            # save results
            savepath = os.path.join(savedir, 'ood_results.json')
            r = {}
            if os.path.isfile(savepath):
                r = json.load(open(savepath, 'r'))

            r[f'round{len(r)}'] = results
            
            json.dump(r, open(savepath, 'w'), cls=MyEncoder, indent='\t')
        else:
            return results
        
class Detector:
    def __init__(
        self, 
        modelname: str,
        num_classes: int,
        epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        num_workers: int,
        temperature: float,
        lr: float,
        opt_name: str,
        sched_name: str,
        savedir: str,
        opt_params: dict = {},
        sched_params: dict = {},
        warmup_params: dict = {},
        accelerator = None  
    ):
        
        self.accelerator = accelerator
        _, self.detector = create_model(modelname=modelname, num_classes=num_classes)
        
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.temperature = temperature
        
        self.opt_name = opt_name
        self.lr = lr
        self.opt_params = opt_params
        
        self.sched_name = sched_name
        self.sched_params = sched_params
        self.warmup_params = warmup_params
        
        self.savedir = savedir
        
        self.current_round = len(glob(os.path.join(self.savedir, 'detector*')))
        
    def init_detector(self, device: str):
        detector = deepcopy(self.detector)
        
        return detector.to(device)
        
    def save_model(self, detector):
        # start meta-learning start in second round
        torch.save(detector.state_dict(), os.path.join(self.savedir, f'detector{self.current_round}.pt'))
        self.current_round += 1
        
    def create_dataloader(self, dataset, sample_idx):
        
        if self.steps_per_epoch > 0:
            dataloader = DataLoader(
                dataset     = TrainIterableDataset(dataset=deepcopy(dataset), sample_idx=sample_idx),
                batch_size  = self.batch_size,
                num_workers = self.num_workers,
            )
        else:         
            dataloader = DataLoader(
                dataset     = deepcopy(dataset),
                batch_size  = self.batch_size,
                sampler     = SubsetRandomSampler(indices=sample_idx),
                num_workers = self.num_workers,
            )
        
        return dataloader
        
    def fit_detector(self, detector, dataset, sample_idx, device: str):      
        # create dataloader
        trainloader = self.create_dataloader(
            dataset    = dataset, 
            sample_idx = sample_idx
        )
          
        # create optimizer and scheduler
        optimizer = create_optimizer(opt_name=self.opt_name, model=detector, lr=self.lr, opt_params=self.opt_params)
        scheduler = create_scheduler(
            sched_name    = self.sched_name, 
            optimizer     = optimizer, 
            epochs        = self.epochs, 
            params        = self.sched_params,
            warmup_params = self.warmup_params
        )
        criterion = nn.CrossEntropyLoss()
        
        if self.accelerator != None:
            detector, trainloader, optimizer, scheduler = self.accelerator.prepare(detector, trainloader, optimizer, scheduler)
        
        # train mode on
        detector.train()
        
        desc = '[TRAIN Detector] Loss: {loss:.>6.4f}, Acc: {acc:.3%}, LR: {lr:.3e}'
        p_bar = tqdm(range(self.epochs))
        for epoch in p_bar:
            loss_avg, acc = self.train_detector(
                detector    = detector,
                trainloader = trainloader,
                optimizer   = optimizer,
                criterion   = criterion,
                device      = device    
            )
            
            scheduler.step()
            
            p_bar.set_description(desc=desc.format(loss=loss_avg, acc=acc, lr=optimizer.param_groups[0]['lr']))
        
        # save detector
        self.save_model(detector=detector)
        
        
    def train_detector(self, detector, trainloader, optimizer, criterion, device: str):
        optimizer.zero_grad()
        
        steps_per_epoch = self.steps_per_epoch if self.steps_per_epoch > 0 else len(trainloader)

        total = 0
        correct = 0
        total_loss = 0
        step = 0
        
        desc = 'Loss: {loss:.>6.4f}, Acc: {acc:.3%} [{correct}/{total}]'
        p_bar = tqdm(trainloader, total=steps_per_epoch, leave=False)
        for inputs, targets in p_bar:
            if self.accelerator == None:
                inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = detector(inputs)
            
            outputs /= self.temperature
                
            loss = criterion(outputs, targets)    
            
            if self.accelerator != None:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # loss update
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            # accuracy 
            correct += targets.eq(outputs.argmax(dim=1)).sum().item()
            total += targets.size(0)
            
            # progress
            p_bar.set_description(
                desc=desc.format(
                    loss    = total_loss / (step+1),
                    acc     = correct / total,
                    correct = correct,
                    total   = total
                )
            )
            
            step += 1
            
            if step == steps_per_epoch:
                break
            
        loss_avg = total_loss / steps_per_epoch
        acc = correct / total
        
        return loss_avg, acc