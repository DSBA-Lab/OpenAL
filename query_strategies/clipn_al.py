import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from copy import deepcopy

from query_strategies.metric_learning import create_metric_learning
from .sampler import SubsetSequentialSampler
from .utils import get_target_from_dataset, NoIndent, MyEncoder

class CLIPNAL(nn.Module):
    def __init__(
        self, vis_clf, train_transform, test_transform, num_id_classes: int,
        dataset, batch_size: int, num_workers: int, is_labeled: np.ndarray, is_unlabeled: np.ndarray, 
        savedir: str, use_sim: bool = False, metric_params: dict = {}):
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
        
        self.savedir = savedir
        self.use_sim = use_sim
        self.use_metric_learning = metric_params.get('use', False)
        if self.use_metric_learning:
            self.metric_learning = create_metric_learning(
                method_name     = metric_params['method_name'],
                vis_encoder     = SupConModel(image_encoder=deepcopy(self.vis_clf.image_encoder)), 
                criterion       = nn.CrossEntropyLoss(), 
                epochs          = metric_params['epochs'], 
                train_transform = train_transform, 
                test_transform  = test_transform,
                test_ratio      = metric_params['test_ratio'], 
                opt_name        = metric_params['opt_name'], 
                lr              = metric_params['lr'], 
                savedir         = savedir, 
                seed            = metric_params['seed'], 
                opt_params      = metric_params['opt_params'],
                **metric_params.get('train_params', {})
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
    
        # confusion matrix
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    
        # save results
        savepath = os.path.join(savedir, 'ood_results.json')
        r = {}
        if os.path.isfile(savepath):
            r = json.load(open(savepath, 'r'))

        r[f'round{len(r)}'] = {
            'cm': [NoIndent(elem) for elem in cm.tolist()],
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
        
        json.dump(r, open(savepath, 'w'), cls=MyEncoder, indent='\t')
        
    def check_sim_acc(self, y_pred: np.ndarray, savedir: str):
        
        # get unlabeled targets
        ulb_idx = np.where(self.is_unlabeled==True)[0]
        targets = get_target_from_dataset(self.dataset)
        y_true = targets[ulb_idx]
        
        # get ID and OOD index from unlabeled samples
        id_idx = [i for i, t in enumerate(y_true) if t < self.num_id_classes]
        ood_idx = [i for i, t in enumerate(y_true) if t == self.num_id_classes]
        nb_ulb = len(id_idx) + len(ood_idx)
               
        # calc acc
        id_acc = (y_pred[id_idx] == y_true[id_idx]).sum() / len(id_idx)
    
        # save results
        savepath = os.path.join(savedir, 'sim_results.json')
        r = {}
        if os.path.isfile(savepath):
            r = json.load(open(savepath, 'r'))

        r[f'round{len(r)}'] = {
            'acc': {
                'id' : id_acc, 
            },
            'num_samples': {
                'total' : nb_ulb, 
                'id'    : len(id_idx), 
                'ood'   : len(ood_idx)   
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
            self.metric_learning.fit(
                vis_encoder = vis_encoder,
                dataset     = self.dataset,
                sample_idx  = np.where(self.is_labeled==True)[0],
                device      = device
            )
        else:
            vis_encoder = vis_clf.image_encoder
        
        # get ID logits ('logits_yes') and unlabeled image embeddings
        logits_yes, img_embed_ulb = self.get_unlabeled_features(vis_clf=vis_clf, vis_encoder=vis_encoder, device=device)
                
        # using similarity score per class
        if self.use_sim or self.use_metric_learning:
            img_embed_lb_c = self.get_labeled_cls_features(vis_encoder=vis_encoder, device=device)
            score_c = 100 * img_embed_ulb @ img_embed_lb_c.t() # temperature 100
            logits_yes = logits_yes * score_c.softmax(dim=1)
            
            # check acc
            self.check_sim_acc(y_pred=score_c.argmax(dim=1).cpu().numpy(), savedir=self.savedir)
        
        # get ID and OOD scores
        ood_score = 1-logits_yes.sum(dim=1) # OOD score
        id_score = logits_yes.max(dim=1)[0] # ID score
        
        # if self.use_sim or self.use_metric_learning:
        #     # ood_score /= self.num_id_classes # OOD score 
        #     id_score = logits_yes_sim.max(dim=1)[0] # ID score
        
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
    
    
    
class SupConModel(nn.Module):
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