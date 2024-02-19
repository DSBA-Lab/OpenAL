import os
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from query_strategies.metric_learning import create_metric_learning, MetricModel
from .strategy import Strategy
from .sampler import SubsetSequentialSampler
from .utils import torch_seed, get_target_from_dataset

class CCAL(Strategy):
    def __init__(
        self,
        k: float,
        t: float,
        seed: int, 
        savedir: str, 
        semantic_params: dict = {}, 
        distinctive_params: dict = {}, 
        accelerator = None,
        **init_args
    ):
        
        super(CCAL, self).__init__(**init_args)
        
        self.savedir = savedir
        self.seed = seed
                
        # contrastive-learning for distinctive features
        distinctive_encoder = MetricModel(
            modelname   = distinctive_params['modelname'], 
            pretrained  = distinctive_params['pretrained'],
            simclr_dim  = distinctive_params['simclr_dim'],
            **distinctive_params.get('model_params', {})
        )
        self.distinctive_cl = create_metric_learning(
            method_name      = 'SimCLRCSI',
            vis_encoder      = distinctive_encoder,
            dataname         = self.dataset.dataname,
            img_size         = self.dataset.img_size,
            sim_lambda       = distinctive_params['sim_lambda'],
            epochs           = distinctive_params['epochs'],
            batch_size       = distinctive_params['batch_size'], 
            num_workers      = distinctive_params['num_workers'],
            shift_trans_type = distinctive_params['shift_trans_type'], 
            opt_name         = distinctive_params['opt_name'], 
            opt_params       = distinctive_params.get('opt_params', {}),
            lr               = distinctive_params['lr'], 
            sched_name       = distinctive_params['sched_name'],
            sched_params     = distinctive_params['sched_params'],
            warmup_params    = distinctive_params.get('warmup_params', {}),
            savepath         = os.path.join(savedir, 'distinctive_model.pt'), 
            accelerator      = accelerator,
            seed             = seed, 
        )
        
        # contrastive-learning for semantic features
        semantic_encoder = MetricModel(
            modelname   = semantic_params['modelname'], 
            pretrained  = semantic_params['pretrained'],
            simclr_dim  = semantic_params['simclr_dim'],
            **semantic_params.get('model_params', {})
        )
        self.semantic_cl = create_metric_learning(
            method_name      = 'SimCLR',
            vis_encoder      = semantic_encoder,
            dataname         = self.dataset.dataname,
            img_size         = self.dataset.img_size,
            epochs           = semantic_params['epochs'],
            batch_size       = semantic_params['batch_size'], 
            num_workers      = semantic_params['num_workers'],
            opt_name         = semantic_params['opt_name'], 
            opt_params       = semantic_params.get('opt_params', {}),
            lr               = semantic_params['lr'], 
            sched_name       = semantic_params['sched_name'],
            sched_params     = semantic_params['sched_params'],
            warmup_params    = semantic_params.get('warmup_params', {}),
            savepath         = os.path.join(savedir, 'semantic_model.pt'), 
            accelerator      = accelerator,
            seed             = seed, 
        )
        
        # semantic parameters
        self.k = k
        self.t = t
    
    def query(self, model, **kwargs) -> np.ndarray:
        # device
        device = next(model.parameters()).device
        
        # idx
        labeled_idx = np.where(self.is_labeled==True)[0]
        unlabeled_idx = np.where(self.is_unlabeled==True)[0]
        total_idx = np.r_[
            np.where(self.is_labeled==True)[0], 
            np.where(self.is_unlabeled==True)[0], 
            np.where(self.is_ood==True)[0]
        ]
        
        # fit representation model using CSI
        distinctive_encoder = self.distinctive_cl.init_model(device=device)
        if not os.path.isfile(self.distinctive_cl.savepath):
            self.distinctive_cl.fit(
                vis_encoder = distinctive_encoder,
                dataset     = self.dataset,
                sample_idx  = total_idx,
                device      = device,
            )
            
        semantic_encoder = self.semantic_cl.init_model(device=device)
        if not os.path.isfile(self.semantic_cl.savepath):
            self.semantic_cl.fit(
                vis_encoder = semantic_encoder,
                dataset     = self.dataset,
                sample_idx  = total_idx,
                device      = device,
            )
        
        # get labeled and unlabeled normalized features using distinctive representation model
        # ulb_noramlized_embed_dis ( k_shift x N_ulb x d )
        ulb_noramlized_embed_dis = self.get_features(
            vis_encoder = distinctive_encoder, 
            sample_idx  = unlabeled_idx, 
            cl_class    = self.distinctive_cl,
            device      = device,
            desc        = 'Distinctive Unlabeled'
        )
        
        # lb_normalized_embed_dis  ( k_shift x (N_lb,t) x d) in t list )
        lb_normalized_embed_dis = []
        lb_targets = get_target_from_dataset(dataset=self.dataset)[labeled_idx]
        for t in np.unique(lb_targets):
            lb_normalized_embed_dis_t = self.get_features(
                vis_encoder = distinctive_encoder, 
                sample_idx  = np.where(lb_targets==t)[0],
                cl_class    = self.distinctive_cl,
                device      = device,
                desc        = f'Distinctive Labeled-target{t}'
            )
            lb_normalized_embed_dis.append(lb_normalized_embed_dis_t)
        
        # get labeled and unlabeled normalized features using semantic representation model
        # ulb_noramlized_embed_sem ( k_shift x N_ulb x d )
        ulb_noramlized_embed_sem = self.get_features(
            vis_encoder = semantic_encoder, 
            sample_idx  = unlabeled_idx, 
            cl_class    = self.semantic_cl,
            device      = device,
            desc        = 'Semantic Unlabeled'
        )
        
        # lb_normalized_embed_sem ( k_shift x N_lb x d )
        lb_normalized_embed_sem = self.get_features(
            vis_encoder = semantic_encoder, 
            sample_idx  = labeled_idx, 
            cl_class    = self.semantic_cl,
            device      = device,
            desc        = 'Distinctive Label'
        )
        
        # get distictive scores in each class for unlabeled samples ( N_ulb in t list )
        distinctive_scores = []
        for lb_normalized_embed_dis_t in lb_normalized_embed_dis:    
            distinctive_scores_t = self.get_distinctive_scores(
                ulb_features = ulb_noramlized_embed_dis, 
                lb_features  = lb_normalized_embed_dis_t
            )
            distinctive_scores.append(distinctive_scores_t)

        # get semantic scores and pseudo_labels for unlabeled samples ( semantic_scores: N_ulb, pseudo_labels: N_ulb )
        semantic_scores, pseudo_labels = self.get_semantic_scores(
            ulb_features = ulb_noramlized_embed_sem, 
            lb_features  = lb_normalized_embed_sem
        )
            
        # select query
        select_idx = self.select_query(
            distinctive_scores = distinctive_scores, 
            semantic_scores    = semantic_scores, 
            pseudo_labels      = pseudo_labels
        )
        
        return select_idx
    
    def minmax(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def get_distinctive_scores(self, ulb_features: torch.FloatTensor, lb_features: torch.FloatTensor):
        # t: k_shift, i or j: the number of sample index, d: dimension
        sim = torch.einsum('tid, tjd -> tij', ulb_features, lb_features)
        sim_sort, sim_rank_idx = sim.sort(descending=True, dim=2)

        # difference in cosine similarity between first and second for unlabeled samples
        sim_diff_st_nd = sim_sort.select(dim=2, index=0) - sim_sort.select(dim=2, index=1)
        
        # cosine similarity between first and second
        lb_st_idx = sim_rank_idx.select(dim=2, index=0).unsqueeze(2).repeat(1, 1, lb_features.size(2))
        lb_nd_idx = sim_rank_idx.select(dim=2, index=1).unsqueeze(2).repeat(1, 1, lb_features.size(2))

        lb_st_embed = torch.gather(lb_features, dim=1, index=lb_st_idx)
        lb_nd_embed = torch.gather(lb_features, dim=1, index=lb_nd_idx)
        
        sim_st_nd = torch.einsum('tid, tid -> ti', lb_st_embed, lb_nd_embed)
        
        # distinctive scores
        sim = (sim_diff_st_nd + sim_st_nd).mean(dim=0) # average for transform axis
        distinctive_scores = 1 - self.minmax(x=sim)
        
        return distinctive_scores
    
    
    def get_semantic_scores(self, ulb_features: torch.FloatTensor, lb_features: torch.FloatTensor):
        # t: k_shift, i or j: the number of sample index, d: dimension
        sim = torch.einsum('tid, tjd -> tij', ulb_features, lb_features)
        sim_sort, sim_rank_idx = sim.sort(descending=True, dim=2)
        
        sim = sim_sort.select(dim=2, index=0).mean(dim=0) # average for transform axis
        semantic_scores = self.minmax(x=sim)
        
        # get psuedo labels from labeled targets using maximum cosine similarity index of unlabeled samples
        pseudo_labels_idx = sim_rank_idx.select(dim=2, index=0)[0]
        lb_targets = get_target_from_dataset(dataset=self.dataset)[self.is_labeled]
        pseudo_labels = lb_targets[pseudo_labels_idx]
    
        return semantic_scores, pseudo_labels
    
    
    def select_query(self, distinctive_scores: list, semantic_scores: torch.FloatTensor, pseudo_labels: np.ndarray):
        
        # get n_query per class
        n_query_cls = self.get_n_query_cls()
            
        # get selected index for query
        select_idx = []
        for t_idx, distinctive_scores_t in enumerate(distinctive_scores):
            query_scores = self.get_query_scores(distinctive_scores=distinctive_scores_t, semantic_scores=semantic_scores)
            
            # only consider query score for psuedo labels t
            query_scores[pseudo_labels != t_idx] = -1
            
            _, rank_idx = torch.topk(query_scores, n_query_cls[t_idx])
            select_idx.append(rank_idx)
        
        select_idx = torch.cat(select_idx, dim=0)
        
        return select_idx
    

    def get_query_scores(self, distinctive_scores: torch.FloatTensor, semantic_scores: torch.FloatTensor):
        query_scores = torch.tanh(self.k * (semantic_scores - self.t)) + distinctive_scores
        return query_scores

        
    def get_n_query_cls(self):
        n_query_cls = [int(self.n_query / self.num_id_class) for _ in range(self.num_id_class)]
        
        # if n_query does not be divided by num_id_class, then add one in each class for the remainder and shuffle
        if sum(n_query_cls) < self.n_query:
            remain = self.n_query - sum(n_query_cls)
            for i in range(remain):
                n_query_cls[i] += 1
            np.random.shuffle(n_query_cls)
            
        return n_query_cls
    
    
    def get_features(self, vis_encoder, sample_idx, cl_class, device: str, desc: str = ''):
        torch_seed(self.seed)
        
        dataset = deepcopy(self.dataset)
        dataset.transform = self.test_transform
        
        dataloader = DataLoader(
            dataset     = dataset, 
            sampler     = SubsetSequentialSampler(indices=sample_idx), 
            batch_size  = self.batch_size, 
            num_workers = self.num_workers
        )
        
        cl_class.hflip.to(device)
        cl_class.simclr_aug.to(device)

        k_shift = getattr(cl_class, 'k_shift', 1)
        
        vis_encoder.eval()
        
        features = []
        p_bar = tqdm(dataloader, desc=f'Get outputs [{desc} Embed]')
        with torch.no_grad():
            for idx, (images, targets) in enumerate(p_bar):          
                # augment images                
                images = images.to(device)
                if k_shift > 1:
                    images = torch.cat([cl_class.shift_transform(cl_class.hflip(images), k) for k in range(cl_class.k_shift)])
                images = cl_class.simclr_aug(images)
                
                # outputs
                features_i = vis_encoder(images)['simclr']
                features_i = torch.stack(features_i.chunk(k_shift, dim=0), dim=1).cpu()
                
                features.append(features_i)
                
        features = torch.cat(features, dim=0) # ( N x k_shift x d ) N is the number of sample index
        features = features.permute(1,0,2) # ( k_shift x N x d )
        features = F.normalize(features, dim=2)
        
        return features