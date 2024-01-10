
import numpy as np
import torch
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import mode
from collections import defaultdict
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from timm.models import FeatureDictNet
from sklearn.metrics import silhouette_score

from .sampler import SubsetSequentialSampler, SubsetWeightedRandomSampler
from .utils import get_target_from_dataset
from .tta import TTA

class Strategy:
    def __init__(
        self, model, n_query: int, dataset: Dataset, test_transform, is_labeled: np.ndarray, 
        batch_size: int, num_workers: int, sampler_name: str, n_subset: int = 0, 
        tta_agg: str = None, tta_params: dict = None, interval_type: str = 'top', resampler_params: dict = None):
        
        # model
        self.model = model
        
        # for AL
        self.n_query = n_query
        
        if n_subset > 0:
            if isinstance(n_subset, float):
                assert (n_subset < 1.) and (n_subset > 0.), 'the subset ratio must be between 0. and 1.'
            elif isinstance(n_subset, int):
                assert n_subset > n_query, 'the number of subset must larger than the number of query.'
        self.n_subset = n_subset
        
        # for labeled index
        self.is_labeled = is_labeled 
        
        # for datasets
        self.dataset = dataset
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler_name = sampler_name
        
        # test time augmentation
        self.tta = None
        if tta_params != None:
            self.tta = TTA(agg=tta_agg, params=tta_params)

        # interval type
        self.interval_type = interval_type

        # resampler
        self.use_resampler = False    
        if resampler_params != None:
            for k, v in resampler_params.items():
                setattr(self, k, v)
            self.use_resampler = True

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def init_model(self):
        return deepcopy(self.model)
        
    def loss_fn(self, outputs, targets):
        if isinstance(outputs, dict):
            outputs = outputs['logits']
        return self.criterion(outputs, targets)
        
    def query(self):
        raise NotImplementedError
    
    def update(self, query_idx: np.ndarray):
        self.is_labeled[query_idx] = True
        
    def get_trainloader(self) -> DataLoader:
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = self.select_sampler(indices=np.where(self.is_labeled==True)[0]),
            num_workers = self.num_workers
        )
        
        return dataloader
    
    def select_sampler(self, indices: np.ndarray):
        if self.sampler_name == 'SubsetRandomSampler':
            # sampler
            sampler = SubsetRandomSampler(indices=indices)
        elif self.sampler_name == 'SubsetWeightedRandomSampler':
            # get labels
            labels = get_target_from_dataset(self.dataset)[self.is_labeled]
            
            # calculate weights per samples
            _, labels_cnt = np.unique(labels, return_counts=True)
            labels_weights = 1 - (labels_cnt / labels_cnt.sum())
            weights = [labels_weights[i] for i in labels]
            
            # sampler
            sampler = SubsetWeightedRandomSampler(indices=indices, weights=weights)
            
        return sampler

    def subset_sampling(self, indices: np.ndarray, n_subset: int or float):
        # define subset
        if isinstance(n_subset, int):
            subset_indices = np.random.choice(indices, size=n_subset, replace=False)
        elif isinstance(n_subset, float):
            n_subset = int(len(indices)*n_subset)
            subset_indices = np.random.choice(indices, size=n_subset, replace=False)
            
        return subset_indices

    def get_unlabeled_idx(self):
        
        # get unlabeled index
        unlabeled_idx = np.where(self.is_labeled==False)[0]
            
        # subsampling
        if self.n_subset > 0:
            unlabeled_idx = self.subset_sampling(indices=unlabeled_idx, n_subset=self.n_subset)
            
        return unlabeled_idx

    def extract_outputs(self, model, sample_idx: np.ndarray, num_mcdropout: int = 0,
                        return_probs: bool = True, return_embed: bool = False, return_labels: bool = False) -> torch.Tensor or dict:
        
        # copy dataset
        dataset = deepcopy(self.dataset)
        dataset.transform = self.test_transform
        
        # define sampler
        sampler = SubsetSequentialSampler(indices=sample_idx)
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        # inference
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            if num_mcdropout > 0:
                # results type is torch.Tensor
                results = self.mcdrop_outputs(
                    model         = model, 
                    dataloader    = dataloader, 
                    device        = device, 
                    num_mcdropout = num_mcdropout
                )
            else:
                # results type is dict
                results = self.get_outputs(
                    model         = model,
                    dataloader    = dataloader,
                    device        = device,
                    return_probs  = return_probs,
                    return_embed  = return_embed,
                    return_labels = return_labels
                )
                
        return results
    
    
    def get_outputs(
        self, model, dataloader, device: str, 
        return_probs: bool = True, return_embed: bool = False, return_labels: bool = False) -> dict:

        if self.tta != None:
            assert return_embed == False, 'if you use TTA, return_embed should be False'
        
        # predict
        results = defaultdict(list)
        
        output_desc = []
        if return_probs:
            output_desc.append('Probs')
        if return_embed:
            output_desc.append('Embed')
            # feature extraction
            if getattr(self, 'resampler_use_fe', None):
                model_fe = FeatureDictNet(model)
                results['embed'] = defaultdict(list)
        if return_labels:
            output_desc.append('Labels')
            
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc=f"Get outputs [{','.join(output_desc)}]", leave=False):
                if len(batch) == 2:
                    # for labeled dataset that contains labels
                    inputs, labels = batch
                else:
                    # for unlabeled dataset that does not contain labels
                    inputs = batch
                
                # return labels    
                if return_labels:
                    results['labels'].append(labels.cpu())
                    
                # return embedings            
                if return_embed:
                    if getattr(self, 'resampler_use_fe', None):
                        embed = model_fe(inputs.to(device))
                        for l_idx, f_i in embed.items():
                            results['embed'][l_idx].append(f_i.cpu())
                        
                        # for forward
                        embed = inputs
                        forward_func = 'forward'
                    else:
                        embed = model.forward_features(inputs.to(device))
                        results['embed'].append(embed.cpu())
                        forward_func = 'forward_head'
                else:
                    embed = inputs
                    forward_func = 'forward'
                
                # return probs
                if return_probs:        
                    # forward head
                    if self.tta != None:
                        outputs = self.tta_outputs(model=model, inputs=inputs, device=device)
                    else:  
                        outputs = model.__getattribute__(forward_func)(embed.to(device))
                        if isinstance(outputs, dict):
                            outputs = outputs['logits']
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                    results['probs'].append(outputs.cpu())
                
        # stack
        for k, v in results.items():
            if k == 'labels':
                results[k] = torch.hstack(v)
            else:
                if k == 'embed' and getattr(self, 'resampler_use_fe', None):
                    for l_idx, f_i in results['embed'].items():
                        results['embed'][l_idx] = torch.vstack(f_i)
                else:
                    results[k] = torch.vstack(v)
    
        return results
    
    def tta_outputs(self, model, inputs: torch.Tensor, device: str):
        with torch.no_grad():
            inputs_augs = [inputs] + self.tta(inputs)
            outputs_augs = []
            for inputs_i in inputs_augs:
                outputs = model(inputs_i.to(device))
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                outputs_augs.append(outputs)
                
            outputs_agg = self.tta.aggregate(outputs=outputs_augs)
            
        return outputs_agg
        
    def mcdrop_outputs(self, model, dataloader, device: str, num_mcdropout: int) -> torch.Tensor:
        # predict
        model.train()
        
        mc_probs = []
        # iteration for the number of MC Dropout
        for _ in tqdm(range(num_mcdropout), desc='MC Dropout', leave=False):
            probs = self.get_outputs(model=model, dataloader=dataloader, device=device)['probs']
            mc_probs.append(probs)
            
        return torch.stack(mc_probs)
    
    
    def query_interval(self, unlabeled_idx, model):
        N = len(unlabeled_idx)
        
        if self.interval_type == 'uniform':
            interval = int(N / self.n_query)
            q_idx = list(range(0, N, interval))
            
        elif self.interval_type == 'top':
            q_idx = list(range(0, self.n_query))
            
        elif self.interval_type == 'bottom':
            q_idx = list(range(N-1, N-self.n_query-1, -1))
        
        elif self.interval_type == 'exp':
            q_idx = self.exp_interval(N=N)
            
        elif self.interval_type == 'silhouette_linear':
            q_idx = self.silhouette_linear_interval(N=N, model=model)
            
        elif self.interval_type == 'silhouette_exp':
            q_idx = self.silhouette_exp_interval(N=N, model=model)
        
        return q_idx
    
    
    def get_silhouette_score(self, model):
        # predict probability and embedding on labeled dataset
        lb_embed = self.extract_outputs(
            model        = model, 
            sample_idx   = np.where(self.is_labeled==True)[0], 
            return_probs = False,
            return_embed = True
        )['embed']
        lb_embed = self.pooling_embedding(x=lb_embed)
        labels = get_target_from_dataset(self.dataset)[self.is_labeled]
        
        sil = silhouette_score(X=lb_embed, labels=labels)
        
        return sil
    
    
    def silhouette_linear_interval(self, N, model):
        sil = self.get_silhouette_score(model=model)
        
        if sil < 0:
            interval = int(N / self.n_query)
            q_idx = list(range(0, N, interval))
        else:
            q_idx = self.linear_interval(N=N, sil=sil)
            
        return q_idx
    
    def silhouette_exp_interval(self, N, model):
        sil = self.get_silhouette_score(model=model)
        
        if sil < 0:
            interval = int(N / self.n_query)
            q_idx = list(range(0, N, interval))
        else:
            q_idx = self.exp_interval(N=N, sil=sil)
            
        return q_idx
    
    
    def linear_interval(self, N, sil, max_sil = 0.1):
        if sil > max_sil:
            q_idx = np.arange(self.n_query)
        else:
            s = int(N/self.n_query) - 1
            all_n = int(N - ((sil/max_sil) * s * self.n_query))
            
            interval = int(all_n / self.n_query)
            q_idx = list(range(0, all_n, interval))[:self.n_query]
        
        return q_idx
    
    def exp_interval(self, N, sil, max_sil = 0.1):
        if sil > max_sil:
            q_idx = np.arange(self.n_query)
        else:
            s = int(N/self.n_query) - 1
            all_n = int(N - ((sil/max_sil) * s * self.n_query))
        
            if all_n == self.n_query:
                return np.arange(self.n_query)
            else:
                q_idx = []
                scale = 0.01
                
                while len(q_idx) < self.n_query:
                    linear_spaced = np.linspace(0, 1, self.n_query)
                    exponential_spaced = np.exp(linear_spaced/scale) / scale
                    normalized_samples = self.minmax(exponential_spaced) * (all_n-1)
                    q_idx = np.unique(np.round(normalized_samples).astype(int))
                    
                    scale += 0.01
            
        return q_idx
    
    def minmax(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    
    def resampler(self, model):
        # get labeled index
        labeled_idx = np.where(self.is_labeled==True)[0]
            
        # get embeddings
        embed = self.extract_outputs(
            model        = model, 
            sample_idx   = labeled_idx,
            return_probs = False,
            return_embed = True
        )['embed']
        
        if self.resampler_use_fe:
            scores = np.zeros((len(embed), len(labeled_idx)))
            for i, (l_idx, f_i) in enumerate(embed.items()):
                embed[l_idx] = self.pooling_embedding(x=f_i)
                scores[i] = self._resampler(embed[l_idx])
            
            scores = scores.mean(axis=0)
                
        else:
            embed = self.pooling_embedding(x=embed)
        
            if self.resampler_method == 'lid':
                # get scores of intrinsic dimensionality
                scores = self.lid_resampler(embed)
            elif self.resampler_method == 'random':
                scores = self.get_random_selection(embed=embed, k=self.resampler_k)

            elif self.resampler_method == 'entropy':
                scores = self.get_entropy(embed=embed, k=self.resampler_k)
        
        # sampling
        if self.resampler_order == 'top':
            re_idx = np.where(scores < self.resampler_threshold)[0]
        elif self.resampler_order == 'bottom':
            re_idx = np.where(scores > self.resampler_threshold)[0]
                
        relabeled_idx = labeled_idx[re_idx]
        is_labeled = np.zeros_like(self.is_labeled, dtype=bool)
        is_labeled[relabeled_idx] = True
        
        # update 'is_labeled'
        self.is_labeled = is_labeled
        
    
    def lid_resampler(self, embed):
        # get scores of intrinsic dimensionality
        if self.resampler_target == 'class':
            # ============
            # by target
            # ============
            labeled_targets = get_target_from_dataset(self.dataset)[self.is_labeled]
            scores = np.zeros(len(embed))
            for c in np.unique(labeled_targets):
                c_idx = np.where(labeled_targets==c)[0]
                
                # sampling k
                k_c = len(embed[c_idx]) if self.resampler_k == -1 else self.resampler_k
            
                scores[c_idx] = self.get_lid_estimation(embed=embed[c_idx], k=k_c)        
            
            if not self.resampler_lid_product:
                scores = self.minmax(scores)
        
        elif self.resampler_target == 'all':
            # ============
            # all
            # ============
            scores = self.get_lid_estimation(embed=embed, k=len(embed) if self.resampler_k == -1 else self.resampler_k)
            
            if not self.resampler_lid_product:
                scores = self.minmax(scores)
        
        return scores
    
    def get_entropy(self, embed, k: int):
        dist = cdist(embed, embed)
        dist = np.apply_along_axis(np.argsort, axis=1, arr=dist)[:, 1:k+1]

        def entropy(x):
            _, cnt = np.unique(x, return_counts=True)
            prob = cnt/cnt.sum()

            return -(prob * np.log2(prob)).sum()

        labeled_targets = get_target_from_dataset(self.dataset)[self.is_labeled]
        neighbor_labels = labeled_targets[dist]
        scores = np.apply_along_axis(entropy, axis=1, arr=neighbor_labels)
        
        neighbor_mode = mode(neighbor_labels, axis=1)[0][:,0]

        match_idx = np.where((labeled_targets == neighbor_mode)==True)[0]   
        
        scores[match_idx] = np.inf
        
        return scores
    
    
    def get_lid_estimation(self, embed, k: int):
        embed = np.asarray(embed, dtype=np.float32)

        k = min(k, len(embed)-1)
        dist = cdist(embed, embed)
        dist = np.apply_along_axis(np.sort, axis=1, arr=dist)[:,1:k+1]
        
        if self.resampler_lid_product:
            f = lambda v: np.prod(v/v[-1])
            mle = np.apply_along_axis(f, axis=1, arr=dist)
        else:
            f = lambda v: - k / np.sum(np.log(v/v[-1]))
            mle = np.apply_along_axis(f, axis=1, arr=dist)
        return mle
    
    def get_random_selection(self, embed, k: int):
        select_idx = np.random.choice(len(embed), k, replace=False)
        
        scores = np.zeros(len(embed))
        scores[select_idx] = 1.
        
        return scores        
    
    def pooling_embedding(self, x):
        '''
        dim : Target dimension for pooling 
        mean : Average Pooling 
        '''
        # TODO modelname
        # if np.argmax(list(x.shape)) == 1:   # x shape : NCHW  -> for Resnet 
        #     dim = (2,3)
        #     emb = x.mean(dim)
        # elif np.argmax(list(x.shape)) == 3: # x shape : NHWC -> for Swin-Transformer 
        #     dim = (1,2)
        #     emb = x.mean(dim)
        # elif len(x.shape) == 3: # x shape : NTC -> for ViT 
        #     emb = x[:,0,:]      # cls token 
        
        emb = x.mean((2,3))
        return emb 