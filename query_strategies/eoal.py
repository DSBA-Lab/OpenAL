import os
import numpy as np
from glob import glob
from copy import deepcopy
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finch import FINCH

from models import create_model
from .sampler import SubsetSequentialSampler
from .strategy import Strategy
from .utils import TrainIterableDataset, IndexDataset
from .optims import create_optimizer
from .scheds import create_scheduler

class EOAL(Strategy):
    def __init__(
        self, 
        savedir: str,
        w_unk_cls: int = 1,
        w_ent: float = 1.,
        pareta_alpha: float = 0.8,
        reg_w: float = 0.1,
        train_params: dict = {},
        binary_clf_params: dict = {},
        detector_params: dict = {},
        accelerator = None,
        **init_params
    ):
        
        super(EOAL, self).__init__(**init_params)
        
        self.savedir = savedir
        self.accelerator = accelerator
        
        # loss weights
        self.w_ent = w_ent
        self.pareta_alpha = pareta_alpha
        self.reg_w = reg_w
        
        # FINCH
        self.w_unk_cls = w_unk_cls
        
        # training: epochs, steps_per_epoch, batch_size, temperature
        self.train_params = train_params
        self.train_params['steps_per_epoch'] = self.train_params.get('steps_per_epoch', 0)
        
        # binary classifier
        self.bc_clf = BCClassifier(
            num_classes   = self.num_id_class,
            epochs        = self.train_params['epochs'],
            lr            = binary_clf_params['lr'],
            opt_name      = binary_clf_params['opt_name'],
            sched_name    = binary_clf_params['sched_name'],
            savedir       = self.savedir,
            opt_params    = binary_clf_params['opt_params'],
            sched_params  = binary_clf_params['sched_params'],
            warmup_params = binary_clf_params.get('warmup_params', {}),
        )
        
        # detector
        self.detect = Detector(
            modelname       = detector_params['modelname'],
            num_classes     = self.num_id_class+1,
            epochs          = self.train_params['epochs'],
            lr              = detector_params['lr'],
            opt_name        = detector_params['opt_name'],
            sched_name      = detector_params['sched_name'],
            savedir         = self.savedir,
            opt_params      = detector_params['opt_params'],
            sched_params    = detector_params['sched_params'],
            warmup_params   = detector_params.get('warmup_params', {}),
        )
        
        
    def query(self, model, **kwargs) -> np.ndarray:
        # whether first round or not
        have_ood = True if (self.is_ood==True).sum() == 0 else False
        
        unlabeled_idx = self.get_unlabeled_idx()
        
        # device
        device = next(model.parameters()).device
        
        # training detector and binary classifier    
        detector, bc_classifier, ood_cluster_info = self.fit(device=device, have_ood=have_ood)
        
        # get detector's predictions, features, and binary classifiers' outputs
        outputs = self.extract_outputs(
            detector      = detector,
            bc_classifier = bc_classifier,
            sample_idx    = unlabeled_idx
        )
        
        # closed-set entropy
        scores_c = self.closed_set_entropy(outputs_bc=outputs['outputs_bc'])
        
        # entropy scores
        if have_ood:
            scores = scores_c
        else:
            # distance-based entropy
            scores_d = self.distance_entropy(features=outputs['features'], ood_cluster_center=ood_cluster_info['centers'])

            # scores
            scores = scores_c - scores_d
        
        # diversity
        selected_id_idx = self.diversity(
            scores    = scores,
            features  = outputs['features'].cpu().numpy(),
            preds_det = outputs['pred_det']
        )
        
        # select query index from unlabeled index
        select_idx = unlabeled_idx[selected_id_idx]
        
        
        return select_idx
    
    def diversity(self, scores: torch.Tensor, features: np.ndarray, preds_det: np.ndarray):
        # clustering features as predicted ID
        features_id_pred = features[preds_det < self.num_id_class]
        labels_c, num_clust, _ = FINCH(features_id_pred, req_clust=self.num_id_class, verbose=True)
        
        # find optimal partition
        partition_idx = 0
        while num_clust[partition_idx] > self.n_query:
            partition_idx += 1
            
        # get cluster labels and number of clusters in the selected partition
        cluster_labels = labels_c[:, partition_idx]
        num_clusters = num_clust[partition_idx]
        
        # number of query per cluster
        nb_pred_id_samples = sum(preds_det < self.num_id_class)
        rem = min(self.n_query, nb_pred_id_samples)
        num_per_cluster = int(rem/num_clusters)
        
        # select query
        pred_id_idx = np.where(preds_det < self.num_id_class)[0]
        scores_pred_id = scores[preds_det < self.num_id_class]
        selected_id_idx = []

        nb_selected_query_per_cls = np.zeros(num_clusters, dtype=np.int)
        while rem > 0:
            print("Remaining Budget to Sample:  ", rem)
            for cls in range(num_clusters):
                scores_cls = scores_pred_id[cluster_labels == cls]
                pred_id_idx_cls = pred_id_idx[cluster_labels == cls]
                
                # select size
                if rem >= num_per_cluster:
                    select_size = min(num_per_cluster, len(scores_cls))
                else:
                    select_size = min(rem, len(scores_cls))
                
                # slicing index
                start_idx = nb_selected_query_per_cls[cls]
                end_idx = start_idx+select_size
                
                score_rank_cls = scores_cls.sort()[1][start_idx:end_idx]
                nb_selected_query_per_cls[cls] += len(score_rank_cls)
                
                # substract selected size
                rem -= select_size
                
                # stack selected idx
                selected_id_idx_cls = pred_id_idx_cls[score_rank_cls]
                
                if isinstance(selected_id_idx_cls, np.int64):
                    selected_id_idx_cls = [selected_id_idx_cls]
                    
                if isinstance(selected_id_idx_cls, np.ndarray) and (len(selected_id_idx_cls) == 0):
                    selected_id_idx_cls = []
                
                selected_id_idx.extend(selected_id_idx_cls)
                
        return selected_id_idx
    
    def closed_set_entropy(self, outputs_bc) -> np.ndarray:
        assert len(outputs_bc.size()) == 3
        assert outputs_bc.size(1) == 2
        
        outputs_bc = F.softmax(outputs_bc, 1)
        scores_c = torch.mean(torch.sum(-outputs_bc * torch.log(outputs_bc + 1e-8), 1), 1)
        
        scores_c = scores_c.cpu()
        
        return scores_c
    
    def distance_entropy(self, features, ood_cluster_center) -> np.ndarray:
        dists = torch.cdist(features, ood_cluster_center)
        q = torch.softmax(-dists, dim=1)
        scores_d = -torch.sum(q*torch.log(q+1e-20), 1)
        scores_d = scores_d / np.log(len(ood_cluster_center))
        
        scores_d = scores_d.cpu()
        
        return scores_d
    
    def extract_outputs(self, detector, bc_classifier, sample_idx: np.ndarray):
        
        # unlabeled dataloader
        dataloader = self.create_dataloader(
            dataset    = self.dataset,
            sample_idx = sample_idx,
            is_train   = False
        )
        if self.accelerator != None:
            dataloader = self.accelerator.prepare(dataloader)
        
        # inference
        device = next(detector.parameters()).device
        detector.eval()
        bc_classifier.eval()
        
        # results type is dict
        results = self.get_outputs(
            detector      = detector,
            bc_classifier = bc_classifier,
            dataloader    = dataloader,
            device        = device,
        )
            
        return results
        
    def get_outputs(self, detector, bc_classifier, dataloader, device: str):
        all_preds_det = []
        all_features = []
        all_outputs_bc = []
        
        with torch.no_grad():
            for _, inputs, targets in tqdm(dataloader, total=len(dataloader), desc='Get outputs'):
                if self.accelerator == None:
                    inputs = inputs.to(device)
                    
                # detector
                features = detector.forward_features(inputs)
                outputs_det = detector.forward_head(features)
                preds_det = F.softmax(outputs_det, dim=1).max(dim=1)[1]
                
                # binary classifier
                features = detector.global_pool(features)
                outputs_bc = bc_classifier(features)
                
                # outputs
                all_preds_det.extend(preds_det.cpu().numpy())
                all_features.append(features)
                all_outputs_bc.append(outputs_bc)
            
        
        all_preds_det = np.asarray(all_preds_det)
        all_features = torch.cat(all_features, dim=0)
        all_outputs_bc = torch.cat(all_outputs_bc, dim=0)
            
        return {
            'pred_det'   : all_preds_det,
            'features'   : all_features,
            'outputs_bc' : all_outputs_bc
        }
            
        
    
    def create_dataloader(self, dataset, sample_idx, is_train: bool = True):
        dataset = deepcopy(dataset)
        
        if is_train:
            if self.train_params['steps_per_epoch'] > 0:
                dataloader = DataLoader(
                    dataset     = TrainIterableDataset(dataset=dataset, sample_idx=sample_idx, return_index=True),
                    batch_size  = self.train_params['batch_size'],
                    num_workers = self.num_workers,
                )
            else:         
                dataloader = DataLoader(
                    dataset     = IndexDataset(dataset=dataset),
                    batch_size  = self.train_params['batch_size'],
                    sampler     = self.select_sampler(indices=sample_idx),
                    num_workers = self.num_workers,
                )
        else:
            dataset.transform = self.test_transform
            dataloader = DataLoader(
                dataset     = IndexDataset(dataset=dataset),
                batch_size  = self.train_params['batch_size'],
                sampler     = SubsetSequentialSampler(indices=sample_idx),
                num_workers = self.num_workers,
            )
        
        return dataloader
    
    def fit(self, device: str, have_ood: bool = False):    
        # sample idx is labeled. If current round is not first, then use with ood samples
        sample_idx = np.where(self.is_labeled==True)[0]
        if not have_ood:
            ood_idx = np.where(self.is_ood==True)[0]
            sample_idx = np.r_[sample_idx, ood_idx]
        
        # create trainloader
        trainloader = self.create_dataloader(dataset=self.dataset, sample_idx=sample_idx, is_train=True)
        
        criterion = nn.CrossEntropyLoss()
        
        # init detector
        detector = self.detect.init_model(device=device)
        optimizer_det, scheduler_det = self.detect.compile(detector=detector)
        
        # init binary classifier
        bc_classifier = self.bc_clf.init_model(in_features=detector.num_features, device=device)
        optimizer_bc, scheduler_bc = self.bc_clf.compile(bc_classifier=bc_classifier)
        
        if self.accelerator != None:
            detector, bc_classifier, trainloader, optimizer_det, scheduler_det, optimizer_bc, scheduler_bc = self.accelerator.prepare(
                detector, bc_classifier, trainloader, optimizer_det, scheduler_det, optimizer_bc, scheduler_bc
            )
        
        # train mode on
        detector.train()
        bc_classifier.train()
        
        desc = '[TRAIN Detector] Loss: {loss:.>6.4f} ' \
               'Loss-CE: {loss_ce:.>6.4f} Loss-BCE: {loss_bce:.>6.4f} Loss-EM: {loss_em:.>6.4f} Loss_T: {loss_t:.>6.4f} ' \
               'Acc: {acc:.3%} DET-LR: {lr_det:.3e} BC-LR: {lr_bc:.3e}'
        
        p_bar = tqdm(range(self.train_params['epochs']))
        
        for epoch in p_bar:
            # if current round is not first, use ood sample clustering information
            if have_ood:
                ood_cluster_info = None
            else:
                ood_cluster_info = self.ood_clustering(detector, ood_idx=ood_idx, device=device)

            results = self.train(
                detector         = detector,
                bc_classifier    = bc_classifier,
                trainloader      = trainloader,
                criterion        = criterion,
                optimizer_det    = optimizer_det,
                optimizer_bc     = optimizer_bc,
                ood_cluster_info = ood_cluster_info,
                have_ood         = have_ood,
                device           = device
            )
            
            scheduler_det.step()
            scheduler_bc.step()
            
            p_bar.set_description(desc=desc.format(
                loss     = results['loss'],
                loss_ce  = results['loss_ce'],
                loss_bce = results['loss_bce'],
                loss_em  = results['loss_em'],
                loss_t   = results['loss_t'],
                acc      = results['acc'], 
                lr_det   = optimizer_det.param_groups[0]['lr'],
                lr_bc    = optimizer_bc.param_groups[0]['lr']
            ))
        
        # save detector and binary classifier
        self.detect.save_model(detector=detector)
        self.bc_clf.save_model(bc_classifier=bc_classifier)
        
        return detector, bc_classifier, ood_cluster_info
    
    def train(
        self, detector, bc_classifier, trainloader, criterion, optimizer_det, optimizer_bc, 
        ood_cluster_info, have_ood: bool, device: str
    ):
        optimizer_det.zero_grad()
        optimizer_bc.zero_grad()
        
        steps_per_epoch = self.train_params['steps_per_epoch'] if self.train_params['steps_per_epoch'] > 0 else len(trainloader)

        total = 0
        correct = 0
        total_loss = 0
        total_loss_ce = 0
        total_loss_bce = 0
        total_loss_em = 0
        total_loss_t = 0
        step = 0
        
        desc = 'Loss: {loss:.>6.4f} ' \
               'Loss-CE: {loss_ce:.>6.4f} Loss-BCE: {loss_bce:.>6.4f} Loss-EM: {loss_em:.>6.4f} Loss_T: {loss_t:.>6.4f} ' \
               'Acc: {acc:.3%} [{correct}/{total}]'
               
        p_bar = tqdm(trainloader, total=steps_per_epoch, leave=False)
        for idx, inputs, targets in p_bar:

            if self.accelerator == None:
                inputs, targets = inputs.to(device), targets.to(device)
            
            # detector
            features = detector.forward_features(inputs)
            outputs_det = detector.forward_head(features)
            
            # cross entropy loss for detector
            outputs_det /= self.train_params['temperature']
            loss_ce = criterion(outputs_det, targets)
            
            # binary classifier
            features = detector.global_pool(features)
            outputs_bc = bc_classifier(features)
            
            # bce loss for ID and em loss for OOD
            loss_bce, loss_em = self.entropic_bc_loss(
                outputs_bc     = outputs_bc, 
                labels         = targets, 
                pareto_alpha   = self.pareta_alpha, 
                weight         = self.w_ent, 
                num_id_class   = self.num_id_class, 
                have_ood       = have_ood
            )
            
            # tuplet loss for clustering
            if have_ood:
                loss_t = 0
            else:
                ood_cluster_idx = []
                for i in idx.cpu().tolist():
                    if i in ood_cluster_info['index']:
                        idx_c = np.where(ood_cluster_info['index'] == i)[0][0]
                        ood_cluster_idx.append(idx_c)
                
                loss_t = self.reg_loss(
                    ood_features    = features[targets == self.num_id_class],
                    cluster_centers = ood_cluster_info['centers'], 
                    cluster_labels  = ood_cluster_info['labels'][ood_cluster_idx]
                )
                
            # calc loss
            loss = loss_ce + loss_bce + loss_em + (self.reg_w * loss_t)
            
            # gradients
            if self.accelerator != None:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # loss update
            optimizer_det.step()
            optimizer_det.zero_grad()
            
            optimizer_bc.step()
            optimizer_bc.zero_grad()
            
            # total loss
            total_loss += loss.item()
            total_loss_ce += loss_ce.item()
            total_loss_bce += loss_bce.item()
            total_loss_em += loss_em if have_ood else loss_em.item()
            total_loss_t += loss_t if have_ood else loss_t.item()

            # accuracy 
            correct += targets.eq(outputs_det.argmax(dim=1)).sum().item()
            total += targets.size(0)
            
            # progress
            p_bar.set_description(
                desc=desc.format(
                    loss     = total_loss / (step+1),
                    loss_ce  = total_loss_ce / (step+1),
                    loss_bce = total_loss_bce / (step+1),
                    loss_em  = total_loss_em / (step+1),
                    loss_t   = total_loss_t / (step+1),
                    acc      = correct / total,
                    correct  = correct,
                    total    = total
                )
            )
            
            step += 1
            
            if step == steps_per_epoch:
                break
            
        loss_avg = total_loss / steps_per_epoch
        loss_ce_avg = total_loss_ce / steps_per_epoch
        loss_bce_avg = total_loss_bce / steps_per_epoch
        loss_em_avg = total_loss_em / steps_per_epoch
        loss_t_avg = total_loss_t / steps_per_epoch
        acc = correct / total
        
        return {
            'loss'     : loss_avg, 
            'loss_ce'  : loss_ce_avg,
            'loss_bce' : loss_bce_avg,
            'loss_em'  : loss_em_avg,
            'loss_t'   : loss_t_avg,
            'acc'      : acc
        }
    
    def ood_clustering(self, detector, ood_idx: np.ndarray, device: str):
        # eval mode
        detector.eval()
        
        # create unlabeled dataloader
        dataloader = self.create_dataloader(dataset=self.dataset, sample_idx=ood_idx, is_train=False)
        
        if self.accelerator != None:
            dataloader = self.accelerator.prepare(dataloader)
        
        # progress bar
        p_bar = tqdm(dataloader, total=len(dataloader), leave=False, desc='[FINCH CLUSTERING]')
        
        # get features and clustering
        all_features = []
        with torch.no_grad():
            for _, inputs, targets in p_bar:
                if self.accelerator == None:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                # get features and logits(outputs)
                features = detector.forward_features(inputs)
                features = detector.global_pool(features)
                all_features.append(features)
            
        all_features = torch.cat(all_features, dim=0)
            
        # clustering
        self.req_clust = self.num_id_class # set req_clust
        req_c = None
        while not isinstance(req_c, np.ndarray):
            try:
                _, _, req_c = FINCH(all_features.cpu().numpy(), req_clust=self.w_unk_cls*self.req_clust, verbose=False)
            except:
                req_c = None
                self.req_clust -= 1
        
        # get cluster labels       
        cluster_labels = torch.tensor(req_c, device=device)
        
        # get cluster center vectors
        cluster_centers = self.calc_cluster_centers(features=all_features, labels=cluster_labels)
            
        return {
            'centers' : cluster_centers, 
            'labels'  : cluster_labels,
            'index'   : ood_idx
        }
        
    def calc_cluster_centers(self, features, labels):
        _, embed_size = features.size() # number of unlabeled samples x embedding size
        nb_cluster = torch.unique(labels).size(0)
        cluster_centers = torch.zeros(nb_cluster, embed_size, device=features.device)
        
        for c in range(nb_cluster):
            # find index of cluster c
            c_idx = torch.where(labels == c)[0]
            features_c = features[c_idx]
            
            cluster_centers[c] = features_c.mean(dim=0)
        
        return cluster_centers
        
    
    def entropic_bc_loss(self, outputs_bc, labels, pareto_alpha: float, weight: float, num_id_class: int, have_ood: bool = False):
        assert len(outputs_bc.size()) == 3
        assert outputs_bc.size(1) == 2

        outputs_bc = F.softmax(outputs_bc, dim=1)
        label_p = torch.zeros((outputs_bc.size(0), outputs_bc.size(2)+1), device=outputs_bc.device)
        
        label_range = torch.arange(0, outputs_bc.size(0))  
        label_p[label_range, labels] = 1 
        label_n = 1 - label_p
        
        if not have_ood:
            label_p[labels==num_id_class, :] = pareto_alpha/num_id_class
            label_n[labels==num_id_class, :] = pareto_alpha/num_id_class
            
        label_p = label_p[:, :-1]
        label_n = label_n[:, :-1]
        
        if (not have_ood) and (weight != 0):
            open_loss_pos = torch.mean(torch.sum(-torch.log(outputs_bc[labels<num_id_class, 1, :] + 1e-8) * (1 - pareto_alpha) * label_p[labels<num_id_class], 1))
            open_loss_neg = torch.mean(torch.max(-torch.log(outputs_bc[labels<num_id_class, 0, :] + 1e-8) * (1 - pareto_alpha) * label_n[labels<num_id_class], 1)[0]) ##### take max negative alone
            open_loss_pos_ood = torch.mean(torch.sum(-torch.log(outputs_bc[labels==num_id_class, 1, :] + 1e-8) * label_p[labels==num_id_class], 1))
            open_loss_neg_ood = torch.mean(torch.sum(-torch.log(outputs_bc[labels==num_id_class, 0, :] + 1e-8) * label_n[labels==num_id_class], 1))
            
        else:
            open_loss_pos = torch.mean(torch.sum(-torch.log(outputs_bc[:, 1, :] + 1e-8) * (1 - 0) * label_p, 1))
            open_loss_neg = torch.mean(torch.max(-torch.log(outputs_bc[:, 0, :] + 1e-8) * (1 - 0) * label_n, 1)[0]) ##### take max negative alone
            open_loss_pos_ood = 0
            open_loss_neg_ood = 0
            
        loss_bce = (open_loss_pos + open_loss_neg) * 0.5
        loss_em = (open_loss_pos_ood + open_loss_neg_ood) * 0.5
        
        return loss_bce, loss_em

    
    def reg_loss(self, ood_features, cluster_centers, cluster_labels):
        uk_dists = torch.cdist(ood_features, cluster_centers)
        
        true = torch.gather(uk_dists, 1, cluster_labels.long().view(-1, 1)).view(-1)
        non_gt = torch.tensor([[i for i in range(len(cluster_centers)) if cluster_labels[x] != i] for x in range(len(uk_dists))], device=uk_dists.device).long()
        others = torch.gather(uk_dists, 1, non_gt)
        
        intra_loss = torch.mean(true)
        
        inter_loss = torch.exp(-others+true.unsqueeze(1))
        inter_loss = torch.mean(torch.log(1+torch.sum(inter_loss, dim = 1)))
        
        loss_t = 0.1*intra_loss + 1*inter_loss
        
        return loss_t


class Detector:
    def __init__(
        self, 
        modelname: str,
        num_classes: int,
        epochs: int,
        lr: float,
        opt_name: str,
        sched_name: str,
        savedir: str,
        opt_params: dict = {},
        sched_params: dict = {},
        warmup_params: dict = {}  
    ):
        
        _, self.detector = create_model(modelname=modelname, num_classes=num_classes)
        
        self.epochs = epochs
        
        self.opt_name = opt_name
        self.lr = lr
        self.opt_params = opt_params
        
        self.sched_name = sched_name
        self.sched_params = sched_params
        self.warmup_params = warmup_params
        
        self.savedir = savedir
        
        self.current_round = len(glob(os.path.join(self.savedir, 'detector*')))
        
    def init_model(self, device: str):
        detector = deepcopy(self.detector)
        
        return detector.to(device)
        
    def save_model(self, detector):
        # start meta-learning start in second round
        torch.save(detector.state_dict(), os.path.join(self.savedir, f'detector{self.current_round}.pt'))
        self.current_round += 1
        
    def compile(self, detector):
        optimizer = create_optimizer(opt_name=self.opt_name, model=detector, lr=self.lr, opt_params=self.opt_params)
        scheduler = create_scheduler(
            sched_name    = self.sched_name, 
            optimizer     = optimizer, 
            epochs        = self.epochs, 
            params        = self.sched_params,
            warmup_params = self.warmup_params
        )
        
        return optimizer, scheduler

    
class BCClassifier:
    def __init__(
        self, 
        num_classes: int,
        epochs: int,
        lr: float,
        opt_name: str,
        sched_name: str,
        savedir: str,
        opt_params: dict = {},
        sched_params: dict = {},
        warmup_params: dict = {}  
    ):
        
        self.num_classes = num_classes
        
        self.epochs = epochs
        
        self.opt_name = opt_name
        self.lr = lr
        self.opt_params = opt_params
        
        self.sched_name = sched_name
        self.sched_params = sched_params
        self.warmup_params = warmup_params
        
        self.savedir = savedir
        
        self.current_round = len(glob(os.path.join(self.savedir, 'bc_classifier*')))
        
    def init_model(self, in_features: int, device: str):
        bc_classifier = BCModel(in_features=in_features, num_classes=self.num_classes*2)
        
        return bc_classifier.to(device)
        
    def save_model(self, bc_classifier):
        # start meta-learning start in second round
        torch.save(bc_classifier.state_dict(), os.path.join(self.savedir, f'bc_classifier{self.current_round}.pt'))
        self.current_round += 1
        
    def compile(self, bc_classifier):
        optimizer = create_optimizer(opt_name=self.opt_name, model=bc_classifier, lr=self.lr, opt_params=self.opt_params)
        scheduler = create_scheduler(
            sched_name    = self.sched_name, 
            optimizer     = optimizer, 
            epochs        = self.epochs, 
            params        = self.sched_params,
            warmup_params = self.warmup_params
        )
        
        return optimizer, scheduler    

    
class BCModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(BCModel, self).__init__()
        self.clf = nn.Linear(in_features=in_features, out_features=num_classes, bias=False)
        
    def forward(self, x):
        out = self.clf(x)
        
        out = out.view(out.size(0), 2, -1) # batch size x 2 x number of classes
        return out