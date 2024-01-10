import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from copy import deepcopy

from .sampler import SubsetSequentialSampler
from .utils import get_target_from_dataset, torch_seed

class CLIPNAL(nn.Module):
    def __init__(
        self, vis_clf, train_transform, test_transform, num_id_classes: int,
        dataset, batch_size: int, num_workers: int, is_labeled: np.ndarray, is_unlabeled: np.ndarray, 
        use_sim: bool = False, metric_params: dict = {}):
        super().__init__()
        
        self.num_id_classes = num_id_classes
        
        self.vis_clf = vis_clf
        self.train_transform = train_transform
        self.test_transform = test_transform
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.dataset = dataset
        
        self.is_labeled = is_labeled
        self.is_unlabeled = is_unlabeled
        
        self.use_sim = use_sim
        self.use_metric_learning = metric_params.get('use', False)
        if self.use_metric_learning:
            self.metric_learning = MetricLearning(
                vis_encoder     = deepcopy(self.vis_clf.image_encoder), 
                criterion       = nn.CrossEntropyLoss(), 
                X               = self.dataset.data[is_labeled], 
                y               = self.dataset.targets[is_labeled], 
                epochs          = metric_params['epochs'], 
                train_transform = train_transform, 
                test_transform  = test_transform,
                test_ratio      = metric_params['test_ratio'], 
                opt_name        = metric_params['opt_name'], 
                lr              = metric_params['lr'], 
                savedir         = metric_params['savedir'], 
                seed            = metric_params['seed'], 
                opt_params      = metric_params['opt_params']
            )
    
    def init_model(self):
        return deepcopy(self.vis_clf)
    
    
    def check_ood_acc(self, pred_id_ulb_idx: np.ndarray, savedir: str):
        
        # get unlabeled targets
        ulb_idx = np.where(self.is_unlabeled==True)[0]
        targets = get_target_from_dataset(self.dataset)
        ulb_targets = targets[ulb_idx]
        
        # get ID and OOD index from unlabeled samples
        id_idx = [i for i, t in enumerate(ulb_targets) if t < self.num_id_classes]
        ood_idx = [i for i, t in enumerate(ulb_targets) if t == self.num_id_classes]
        nb_ulb = len(id_idx) + len(ood_idx)
        
        # set predictions
        y_pred = np.zeros(len(targets))
        y_pred[pred_id_ulb_idx] = 1 # ID: 1, OOD: 0
        y_pred = y_pred[ulb_idx]
        
        # set targets
        y_true = np.array([1 if t < self.num_id_classes else 0 for t in ulb_targets])
        
        # calc acc
        total_acc = (y_pred == y_true).sum() / nb_ulb
        id_acc = (y_pred[id_idx] == y_true[id_idx]).sum() / len(id_idx)
        ood_acc = (y_pred[ood_idx] == y_true[ood_idx]).sum() / len(ood_idx)
    
        # save results
        savepath = os.path.join(savedir, 'ood_results.json')
        r = {}
        if os.path.isfile(savepath):
            r = json.load(open(savepath, 'r'))

        r[f'round{len(r)}'] = {
            'acc': {
                'total' : total_acc, 
                'id'    : id_acc, 
                'ood'   : ood_acc, 
            },
            'num_samples': {
                'total'    : nb_ulb, 
                'id'     : len(id_idx), 
                'ood'    : len(ood_idx)   
            }
        }
        
        json.dump(r, open(savepath, 'w'), indent='\t')
        
    def update(self, query_idx: np.ndarray):
        # turn on only ID query index
        id_query_idx = self.get_id_query_idx(query_idx=query_idx)
        self.is_labeled[id_query_idx] = True 
        
        # turn off query index
        self.is_unlabeled[query_idx] = False 
        
        return id_query_idx
    
        
    def get_id_query_idx(self, query_idx: np.ndarray):
        targets = get_target_from_dataset(self.dataset)
               
        query_targets = targets[query_idx]
        id_idx = np.where(query_targets < self.num_id_classes)[0]
        
        id_query_idx = query_idx[id_idx]
        
        return id_query_idx
        
        
    def get_dataloader(self, sample_idx: np.ndarray):
        dataset = deepcopy(self.dataset)
        dataset.transform = self.test_transform
        
        sampler = SubsetSequentialSampler(indices=sample_idx)
    
        dataloader = DataLoader(
            dataset     = dataset,
            sampler     = sampler,
            batch_size  = self.batch_size,
            num_workers = self.num_workers
        )
        
        return dataloader
        
    def get_labeled_cls_features(self, vis_encoder, device: str):
        dataloader = self.get_dataloader(sample_idx=np.where(self.is_labeled==True)[0])
        
        img_embed_lb = []
        labels = []
              
        # get labeled image embeddings
        vis_encoder.eval()
        with torch.no_grad():
            for inputs_i, targets_i in tqdm(dataloader, total=len(dataloader), desc='[CLIPN] get labeled cls features', leave=False):
                img_embed_lb_i = vis_encoder(inputs_i.to(device))
                img_embed_lb.append(img_embed_lb_i.cpu())
                labels.extend(targets_i.cpu().tolist())

            labels = np.array(labels)
            img_embed_lb = torch.concat(img_embed_lb)
            img_embed_lb = F.normalize(img_embed_lb, dim=1)
            
        # mean embed of each class
        img_embed_lb_c = []
        for c in np.unique(labels):
            img_embed_lb_c.append(img_embed_lb[np.where(labels == c)[0]].mean(dim=0))

        img_embed_lb_c = torch.vstack(img_embed_lb_c)
        
        return img_embed_lb_c
        
        
    def get_unlabeled_features(self, vis_clf, vis_encoder, device: str):
        dataloader = self.get_dataloader(sample_idx=np.where(self.is_unlabeled==True)[0])
        
        logits = []
        no_logits = []
        img_embed_ulb = []
        
        vis_clf.eval()
        vis_encoder.eval()
        with torch.no_grad():
            for inputs_i, _ in tqdm(dataloader, total=len(dataloader), desc='[CLIPN] get unlabeled features', leave=False):
                logits_i, no_logits_i, _ = vis_clf(inputs_i.to(device))
                img_embed_ulb_i = vis_encoder(inputs_i.to(device))
                
                logits.append(logits_i.cpu())
                no_logits.append(no_logits_i.cpu())
                img_embed_ulb.append(img_embed_ulb_i.cpu())
                
            logits = torch.cat(logits)
            no_logits = torch.cat(no_logits)
            img_embed_ulb = torch.cat(img_embed_ulb)
            
            img_embed_ulb = F.normalize(img_embed_ulb, dim=1)

        prob_yes = torch.stack([logits, no_logits], dim=2).softmax(dim=2)[:,:,0]
        logits_yes = prob_yes * torch.softmax(logits, dim=1)
        
        return logits_yes, img_embed_ulb
    
    def predict_id_idx(self, vis_clf):
        # set device
        device = next(vis_clf.parameters()).device
        
        # set visual encoder
        if self.use_metric_learning:
            vis_encoder = self.metric_learning.init_model(device=device)
            self.metric_learning.fit(vis_encoder=vis_encoder, device=device)
        else:
            vis_encoder = vis_clf.image_encoder
        
        # get ID logits ('logits_yes') and unlabeled image embeddings
        logits_yes, img_embed_ulb = self.get_unlabeled_features(vis_clf=vis_clf, vis_encoder=vis_encoder, device=device)
                
        # using similarity score per class
        if self.use_sim or self.use_metric_learning:
            img_embed_lb_c = self.get_labeled_cls_features(vis_encoder=vis_encoder, device=device)
            score_c = img_embed_ulb @ img_embed_lb_c.t() # temperature 100
            logits_yes_sim = logits_yes * score_c.softmax(dim=1)
        
        # get ID and OOD scores
        ood_score = 1-logits_yes.sum(dim=1) # OOD score
        id_score = logits_yes.max(dim=1)[0] # ID score
        
        if self.use_sim or self.use_metric_learning:
            ood_score /= self.num_id_classes # OOD score 
            id_score = logits_yes_sim.max(dim=1)[0] # ID score
        
        # set probs
        probs_id_ood = torch.zeros(len(logits_yes), 2)
        probs_id_ood[:,0] = ood_score 
        probs_id_ood[:,1] = id_score
        
        # get prediction
        pred = probs_id_ood.argmax(dim=1)
        
        # get ID index
        pred_id_idx = torch.where(pred==1)[0].numpy()
        
        pred_id_ulb_idx = np.where(self.is_unlabeled==True)[0][pred_id_idx]
        
        return pred_id_ulb_idx
        
        

class MetricLearning:
    def __init__(
        self, vis_encoder, criterion, X: np.ndarray, y: np.ndarray, epochs: int, train_transform, test_transform,
        test_ratio: float, opt_name: str, lr: float, savedir: str, seed: int, opt_params: dict = {}):
        
        self.vis_encoder = MetricModel(image_encoder=vis_encoder)
        self.criterion = criterion
        self.optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name]
        self.lr = lr
        self.opt_params = opt_params
        
        self.epochs = epochs
        self.savedir = savedir
        self.seed = seed
        
        self.create_datasets(X=X, y=y, test_ratio=test_ratio, train_transform=train_transform, test_transform=test_transform)

    def init_model(self, device: str):
        return deepcopy(self.vis_encoder).to(device)
        
    def fit(self, vis_encoder, device: str):
        acc = 0
        best_acc = 0
        best_epoch = 0

        # optimizer
        optimizer = self.optimizer(vis_encoder.parameters(), lr=self.lr, **self.opt_params)

        desc = '[Metric Learning] Acc: {acc:.2%}, Best Acc: {best_acc:.2%} (best epoch {best_epoch})'
        p_bar = tqdm(range(self.epochs), total=self.epochs, desc=desc.format(acc=acc, best_acc=best_acc, best_epoch=best_epoch))
        
        for epoch in p_bar:
            self.train(vis_encoder=vis_encoder, optimizer=optimizer, device=device)
            acc = self.test(vis_encoder=vis_encoder, device=device)
            
            if best_acc < acc:
                # best metrics
                best_acc = acc
                best_epoch = epoch
                
                # best weights
                best_weights = deepcopy(vis_encoder.state_dict())
            
            p_bar.set_description(desc=desc.format(acc=acc, best_acc=best_acc, best_epoch=best_epoch))
            
        # save log
        self.save_log(best_epoch=best_epoch, best_acc=best_acc)
        
        # load best weights
        vis_encoder.load_state_dict(best_weights)
    
    def save_log(self, best_epoch: int, best_acc: float):
        savepath = os.path.join(self.savedir, 'metric_learning.json')
        
        # load saved file
        r = {}
        if os.path.isfile(savepath):
            r = json.load(open(savepath, 'r'))
        
        # update
        r[f'round{len(r)}'] = {'best_epoch': best_epoch, 'best_acc': float(best_acc)}
        
        # save results
        json.dump(r, open(savepath, 'w'))
        
        
    
    def create_datasets(self, X: np.ndarray, y: np.ndarray, test_ratio: float, train_transform, test_transform):
        # data split
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=test_ratio, stratify=y)
        
        # set trainset        
        _, train_cls_cnt = np.unique(y[train_idx], return_counts=True)
        train_num_batch = min(train_cls_cnt)

        trainset = MetricDataset(
            num_batchs = train_num_batch,
            data       = X[train_idx], 
            labels     = y[train_idx], 
            transform  = train_transform
        )
        
        # set testset
        _, test_cls_cnt = np.unique(y[test_idx], return_counts=True)
        test_num_batch = min(test_cls_cnt)
        test_repeat = 5

        testset = MetricDataset(
            num_batchs = test_num_batch * test_repeat, 
            data       = X[test_idx], 
            labels     = y[test_idx], 
            transform  = test_transform
        )
        
        # set attributions for trainset and testset
        setattr(self, 'trainset', trainset)
        setattr(self, 'testset', testset)
        
    
    def train(self, vis_encoder, optimizer, device: str):
        total_loss = 0
        
        desc = '[TRAIN] Loss: {loss:>6.4f}'
        p_bar = tqdm(self.trainset, desc=desc.format(loss=total_loss), leave=False)
        
        vis_encoder.train()
        torch_seed(self.seed)
        for idx, (anchor, positive) in enumerate(p_bar):
            if idx == len(self.trainset):
                break

            anchor, positive = anchor.to(device), positive.to(device)

            out_anchor, logit_scale = vis_encoder(anchor, return_logits_scaler=True)
            out_positive = vis_encoder(positive)

            targets_i = torch.arange(anchor.size(0)).to(device)

            similarity = torch.einsum('ae, pe -> ap', out_anchor, out_positive)
            similarity = similarity * logit_scale
            loss = self.criterion(similarity, targets_i)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            p_bar.set_description(desc=desc.format(loss=total_loss/(idx+1)))
            
    
    def test(self, vis_encoder, device: str):
        acc = 0
        total_loss = 0
        
        desc = '[TEST] Avg. Acc: {acc:.3%}, Loss: {loss:>6.4f}'
        p_bar = tqdm(self.testset, desc=desc.format(acc=acc, loss=total_loss), leave=False)
        
        vis_encoder.eval()
        torch_seed(self.seed)
        with torch.no_grad():
            for idx, (anchor, positive) in enumerate(p_bar):
                if idx == len(self.testset):
                    break

                anchor, positive = anchor.to(device), positive.to(device)

                out_anchor, logit_scale = vis_encoder(anchor, return_logits_scaler=True)
                out_positive = vis_encoder(positive)

                targets_i = torch.arange(anchor.size(0)).to(device)

                similarity = torch.einsum('ae, pe -> ap', out_anchor, out_positive)
                similarity = similarity * logit_scale
                loss = self.criterion(similarity, targets_i)

                total_loss += loss.item()

                acc += targets_i.eq(similarity.argmax(dim=1)).sum() / len(targets_i)
                
                p_bar.set_description(desc=desc.format(acc=acc/(idx+1), loss=total_loss/(idx+1)))
    
        return acc/(idx+1)
    
    
class MetricDataset(Dataset):
    def __init__(self, num_batchs, data, labels, transform=None):
        self.num_batchs = num_batchs
        self.data = data
        self.labels = labels
        self.transform = transform
        self.num_classes = len(np.unique(labels))
        self.class_idx = dict([(c, np.where(labels==c)[0]) for c in range(self.num_classes)])
    
    def __len__(self):
        return self.num_batchs
    
    def __getitem__(self, i):
        anchors = []
        positives = []
        
        for c in range(self.num_classes):
            # examples in class y_i
            idx_c = self.class_idx[c]
        
            # random choice anchor index and positive index for anchor
            anchor_idx = random.choice(idx_c)
            positive_idx = random.choice(idx_c)
        
            while positive_idx == anchor_idx:
                positive_idx = random.choice(idx_c)
            
            # select anchor and positive image for anchor
            anchor = self.data[anchor_idx]
            positive = self.data[positive_idx]
            
            if self.transform != None:
                anchor = self.transform(Image.fromarray(anchor))
                positive = self.transform(Image.fromarray(positive))
                
            anchors.append(anchor)
            positives.append(positive)
            
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)

        return [anchors, positives]
    
    
class MetricModel(nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters() 
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        
    def forward(self, x, return_logits_scaler: bool = False):
        out = self.image_encoder(x)
        out = F.normalize(out, dim=1)
        
        if return_logits_scaler:
            return out, self.logit_scale.exp()
        else:
            return out