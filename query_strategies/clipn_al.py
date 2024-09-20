import os
import json
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score
from tqdm.auto import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .open_clipn import load_model
from .strategy import Strategy
from .sampler import SubsetSequentialSampler
from .utils import get_target_from_dataset, NoIndent, MyEncoder
from .factory import create_query_strategy

class CLIPNAL(Strategy):
    def __init__(
            self, 
            model_type, 
            ckp_path, 
            prompt_path, 
            savedir: str, 
            selected_strategy: str, 
            use_sim: bool = False,
            **init_args
        ):
        
        super(CLIPNAL, self).__init__(**init_args)
        
        # make query strategy
        del_keys = ['is_openset', 'is_unlabeled', 'is_ood', 'id_classes']
        for k in del_keys:
            del init_args[k]
            
        self.query_strategy = create_query_strategy(strategy_name=selected_strategy, **init_args)
        
        self.savedir = savedir
        
        # load visual classifier
        vis_clf, process_train, process_test = load_model(
            model_type  = model_type, 
            pre_train   = ckp_path, 
            prompt_path = prompt_path, 
            classes     = self.id_classes
        )
        self.vis_clf = vis_clf
        self.clipn_train_transform = process_train
        self.clipn_test_transform = process_test
        
        # similarity
        self.use_sim = use_sim
    
    def init_model(self):
        return self.query_strategy.init_model()
    
    def init_al_model(self):
        return deepcopy(self.vis_clf)
    
    def loss_fn(self, outputs, targets):
        return self.query_strategy.loss_fn(outputs=outputs, targets=targets)
      
    def query(self, model, **kwargs):
        # device
        device = next(model.parameters()).device
        
        # CLIPN visual classifier
        vis_clf = self.init_al_model().to(device)
        
        id_unlabeled_idx = self.get_unlabeled_idx(vis_clf=vis_clf)
        select_idx = self.query_strategy.query(model=model, unlabeled_idx=id_unlabeled_idx)
        
        return select_idx
    
    
    def get_unlabeled_idx(self, vis_clf):
        
        # get unlabeled index
        unlabeled_idx = self.predict_id_idx(vis_clf=vis_clf)
        self.check_ood_acc(id_pred_idx=unlabeled_idx, ulb_sample_idx=np.where(self.is_unlabeled==True)[0], savedir=self.savedir)
            
        # subsampling
        if self.n_subset > 0:
            unlabeled_idx = self.subset_sampling(indices=unlabeled_idx, n_subset=self.n_subset)
            
        return unlabeled_idx
    
        
    def get_clipn_dataloader(self, sample_idx: np.ndarray):
        dataset = deepcopy(self.dataset)
        dataset.transform = self.clipn_test_transform
        
        sampler = SubsetSequentialSampler(indices=sample_idx)
    
        dataloader = DataLoader(
            dataset     = dataset,
            sampler     = sampler,
            batch_size  = self.batch_size,
            num_workers = self.num_workers
        )
        
        return dataloader
    
    def predict_id_idx(self, vis_clf):
        # set device
        device = next(vis_clf.parameters()).device
        
        outputs = {}
        
        # set visual encoder
        vis_encoder = vis_clf.image_encoder
            
        # find best logit scale
        logit_scale = self.get_logit_scale(
            vis_clf        = vis_clf,
            vis_encoder    = vis_encoder,
            ulb_sample_idx = np.r_[np.where(self.is_labeled==True)[0], np.where(self.is_ood==True)[0]],
            lb_sample_idx  = np.where(self.is_labeled==True)[0],
            device         = device
        )
        
        # get ouputs
        outputs = self.get_logits_and_embeds(
            vis_clf        = vis_clf,
            vis_encoder    = vis_encoder,
            ulb_sample_idx = np.where(self.is_unlabeled==True)[0],
            lb_sample_idx  = np.where(self.is_labeled==True)[0],
            device         = device
        )
        
        # get ID and OOD scores
        probs_id_ood = self.get_id_ood_score(logit_scale=logit_scale, **outputs)
        
        # get prediction
        pred = probs_id_ood.argmax(dim=1)
        
        # get ID index
        pred_id_idx = torch.where(pred==1)[0].numpy()
        pred_id_ulb_idx = np.where(self.is_unlabeled==True)[0][pred_id_idx]
        
        return pred_id_ulb_idx
    
    
    def get_logit_scale(self, vis_clf, vis_encoder, device: str, ulb_sample_idx: np.ndarray, lb_sample_idx: np.ndarray = None):
        outputs = self.get_logits_and_embeds(
            vis_clf        = vis_clf,
            vis_encoder    = vis_encoder,
            ulb_sample_idx = ulb_sample_idx,
            lb_sample_idx  = lb_sample_idx,
            device         = device
        )
        
        best_scale = 0
        best_acc = 0
        
        scale_range = np.arange(0.1, 100.1, 0.1)
        desc = "[CURRENT] scale: {scale:.1f} Acc: {acc:.2%} [BEST] scale: {best_scale:.1f} Acc: {best_acc:.2%}"
        p_bar = tqdm(scale_range, total=len(scale_range), desc=desc.format(scale=0.1, acc=0, best_scale=best_scale, best_acc=best_acc))
        
        for scale in p_bar:
            # get ID and OOD scores
            probs_id_ood = self.get_id_ood_score(logit_scale=scale, **outputs)
            
            # get prediction
            pred = probs_id_ood.argmax(dim=1)
            
            # get ID index
            pred_id_idx = torch.where(pred==1)[0].numpy()
            pred_id_ulb_idx = ulb_sample_idx[pred_id_idx]
        
            # acc check
            results = self.check_ood_acc(id_pred_idx=pred_id_ulb_idx, ulb_sample_idx=ulb_sample_idx)
            acc = (results['acc']['id'] + results['acc']['ood']) / 2
            
            if best_acc < acc:
                best_acc = acc
                best_scale = scale
                
            p_bar.set_description(desc=desc.format(scale=scale, acc=acc, best_scale=best_scale, best_acc=best_acc))
                
        return best_scale
    
    
    def get_id_ood_score(self, logits: torch.FloatTensor, no_logits: torch.FloatTensor, logit_scale: float, score_c: torch.FloatTensor = None):
        prob_yes = (torch.stack([logits, no_logits], dim=2) * logit_scale).softmax(dim=2)[:,:,0]
        logits_yes = prob_yes * torch.softmax(logits * logit_scale, dim=1)
    
        ood_score = 1 - logits_yes.sum(dim=1)
        id_score = logits_yes.max(dim=1)[0]
        
        if score_c != None:
            logits_yes *= (score_c * logit_scale).softmax(dim=1)
            id_score = logits_yes.max(dim=1)[0]
            ood_score /= self.num_id_class
            
        # set probs
        probs_id_ood = torch.zeros(len(logits_yes), 2)
        probs_id_ood[:,0] = ood_score 
        probs_id_ood[:,1] = id_score
        
        return probs_id_ood
        
    
    
    def get_logits_and_embeds(self, vis_clf, vis_encoder, device: str, ulb_sample_idx: np.ndarray, lb_sample_idx: np.ndarray = None):
        
        outputs = {}
        
        # get ID logits ('logits_yes') and unlabeled image embeddings
        ulb_features = self.get_unlabeled_features(vis_clf=vis_clf, vis_encoder=vis_encoder, sample_idx=ulb_sample_idx, device=device)
        outputs['logits'] = ulb_features['logits']
        outputs['no_logits'] = ulb_features['no_logits']
                 
        # using similarity score per class
        if self.use_sim:
            lb_features = self.get_labeled_cls_features(vis_encoder=vis_encoder, sample_idx=lb_sample_idx, device=device)
            score_c = ulb_features['img_embed_ulb'] @ lb_features['img_embed_lb_c'].t()
            outputs['score_c'] = score_c
            
            # check acc
            self.check_sim_acc(y_pred=score_c.argmax(dim=1).cpu().numpy(), ulb_sample_idx=ulb_sample_idx, savedir=self.savedir)
            
        return outputs

    
    def get_unlabeled_features(self, vis_clf, vis_encoder, sample_idx: np.ndarray, device: str):
        dataloader = self.get_clipn_dataloader(sample_idx=sample_idx)
        
        ulb_features = {
            'logits': [],
            'no_logits': [],
            'img_embed_ulb': []
        }
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
                
            ulb_features['logits'] = torch.cat(logits)
            ulb_features['no_logits'] = torch.cat(no_logits)
            ulb_features['img_embed_ulb'] = F.normalize(torch.cat(img_embed_ulb), dim=1)
        
        return ulb_features
    
        
    def get_labeled_cls_features(self, vis_encoder, sample_idx: np.ndarray, device: str):
        dataloader = self.get_clipn_dataloader(sample_idx=sample_idx)
        
        img_embed_lb = []
        labels = []
        
        lb_features = {'img_embed_lb_c': []}
              
        # get labeled image embeddings
        vis_encoder.eval()
        with torch.no_grad():
            for inputs_i, targets_i in tqdm(dataloader, total=len(dataloader), desc='[CLIPN] get labeled cls features', leave=False):
                img_embed_lb_i = vis_encoder(inputs_i.to(device))
                img_embed_lb.append(img_embed_lb_i.cpu())
                labels.extend(targets_i.cpu().tolist())

            labels = np.array(labels)
            img_embed_lb = F.normalize(torch.concat(img_embed_lb), dim=1)
                        
        # mean embed of each class
        img_embed_lb_c = []
        for c in np.unique(labels):
            img_embed_lb_c.append(img_embed_lb[np.where(labels == c)[0]].mean(dim=0))

        lb_features['img_embed_lb_c'] = torch.vstack(img_embed_lb_c)
        
        return lb_features
        
        
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
        
    def check_sim_acc(self, y_pred: np.ndarray, ulb_sample_idx: np.ndarray, savedir: str = None):
        
        # get unlabeled targets
        targets = get_target_from_dataset(self.dataset)
        y_true = targets[ulb_sample_idx]
        
        # get ID and OOD index from unlabeled samples
        id_idx = [i for i, t in enumerate(y_true) if t < self.num_id_class]
        ood_idx = [i for i, t in enumerate(y_true) if t == self.num_id_class]
        nb_ulb = len(id_idx) + len(ood_idx)
               
        # calc acc
        id_acc = (y_pred[id_idx] == y_true[id_idx]).sum() / len(id_idx)
    
        # save results
        if savedir:
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
    
    
    
class SupConModel(nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        # for p in self.image_encoder.parameters():
        #     p.requires_grad = False
            
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