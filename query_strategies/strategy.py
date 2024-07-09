
import numpy as np
import torch
from tqdm.auto import tqdm
from collections import defaultdict
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from .sampler import SubsetSequentialSampler, SubsetWeightedRandomSampler
from .utils import get_target_from_dataset, TrainIterableDataset

class Strategy:
    def __init__(
            self, 
            model, 
            n_query: int, 
            dataset: Dataset, 
            transform, 
            is_labeled: np.ndarray, 
            batch_size: int, 
            num_workers: int, 
            sampler_name: str, 
            steps_per_epoch: int = 0,
            n_subset: int = 0, 
            is_openset: bool = False,
            is_unlabeled: np.ndarray = None, 
            is_ood: np.ndarray = None,
            id_classes: np.ndarray = None,
            interval_type: str = 'top', 
            **kwargs
        ):
        
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
        
        # for open-set params
        self.is_openset = is_openset
        if self.is_openset:
            self.is_unlabeled = is_unlabeled
            self.is_ood = is_ood
            self.id_classes = id_classes
            self.num_id_class = len(id_classes)
                        
        # for datasets
        self.dataset = dataset
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler_name = sampler_name
        self.steps_per_epoch = steps_per_epoch
        
        # interval type
        self.interval_type = interval_type

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def init_model(self):
        return deepcopy(self.model)
        
    def loss_fn(self, outputs, targets):
        if isinstance(outputs, dict):
            outputs = outputs['logits']
        return self.criterion(outputs, targets)
        
    def query(self, model, **kwargs) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = kwargs.get('unlabeled_idx', self.get_unlabeled_idx())
        
        # get score rank
        _, score_rank = self.get_scores(model=model, sample_idx=unlabeled_idx)
        
        q_idx = self.query_interval(unlabeled_idx=unlabeled_idx, model=model)
        select_idx = unlabeled_idx[score_rank[q_idx]]
        
        return select_idx
    
    def update(self, query_idx: np.ndarray):
        if self.is_openset:
            # turn off query index
            self.is_unlabeled[query_idx] = False 
            
            # filtering ID index
            id_query_idx, ood_query_idx = self.get_id_query_idx(query_idx=query_idx)
            self.is_labeled[id_query_idx] = True
            self.is_ood[ood_query_idx] = True
            
            return id_query_idx    
        else:
            self.is_labeled[query_idx] = True        
        
    def get_id_query_idx(self, query_idx: np.ndarray):
        targets = get_target_from_dataset(self.dataset)
               
        query_targets = targets[query_idx]
        id_idx = np.where(query_targets < self.num_id_class)[0]
        ood_idx = np.where(query_targets == self.num_id_class)[0]
        
        id_query_idx = query_idx[id_idx]
        ood_query_idx = query_idx[ood_idx]
        
        return id_query_idx, ood_query_idx
    
        
    def get_trainloader(self) -> DataLoader:
        
        if self.steps_per_epoch > 0:
            dataloader = DataLoader(
                dataset     = TrainIterableDataset(dataset=deepcopy(self.dataset), sample_idx=np.where(self.is_labeled==True)[0]),
                batch_size  = self.batch_size,
                num_workers = self.num_workers,
            )
        elif self.steps_per_epoch == 0:
            dataloader = DataLoader(
                dataset     = self.dataset,
                batch_size  = self.batch_size,
                sampler     = self.select_sampler(indices=np.where(self.is_labeled==True)[0]),
                num_workers = self.num_workers,
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
        if self.is_openset:
            unlabeled_idx = np.where(self.is_unlabeled==True)[0]        
        else:
            unlabeled_idx = np.where(self.is_labeled==False)[0]
            
        # subsampling
        if self.n_subset > 0:
            unlabeled_idx = self.subset_sampling(indices=unlabeled_idx, n_subset=self.n_subset)
            
        return unlabeled_idx

    def extract_outputs(self, model, sample_idx: np.ndarray, num_mcdropout: int = 0,
                        return_probs: bool = True, return_embed: bool = False, return_labels: bool = False) -> torch.Tensor or dict:
        
        # copy dataset
        dataset = deepcopy(self.dataset)
        dataset.transform = self.transform
        
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

        # predict
        results = defaultdict(list)
        
        output_desc = []
        if return_probs:
            output_desc.append('Probs')
        if return_embed:
            output_desc.append('Embed')
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
                    backbone = getattr(model, 'backbone', model)
                    
                    embed = backbone.forward_features(inputs.to(device))
                    results['embed'].append(embed.cpu())
                    forward_func = 'forward_head'
                else:
                    embed = inputs
                    forward_func = 'forward'
                
                # return probs
                if return_probs:        
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
                results[k] = torch.vstack(v)
    
        return results
        
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

        if self.interval_type == 'top':
            q_idx = list(range(0, self.n_query))
        
        return q_idx     
    
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