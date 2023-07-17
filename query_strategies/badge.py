import numpy as np
from scipy import stats
from copy import deepcopy as deepcopy
import pdb 

import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

from .strategy import Strategy,SubsetSequentialSampler

class BADGE(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int, num_mcdropout: int = 10):
        
        super(BADGE, self).__init__(
            model       = model,
            n_query     = n_query, 
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
        self.num_mcdropout = num_mcdropout
        
        
    def get_grad_embedding(self, model, n_subset: int = None) -> torch.Tensor:
        # define sampler
        unlabeled_idx = np.where(self.labeled_idx==False)[0]
        sampler = SubsetSequentialSampler(
            indices = self.subset_sampling(indices=unlabeled_idx, n_subset=n_subset) if n_subset else unlabeled_idx
        )
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        embDim = model.num_features
        device = next(model.parameters()).device 
        nLab = len(dataloader.dataset.label_info['label'].unique()) #라벨 수 : 4
        embedding = np.zeros([len(sampler), embDim * nLab])
        
        model.eval()
        with torch.no_grad():
            for idx, (imgs,y) in enumerate(dataloader):
                idxs = np.arange(idx* dataloader.batch_size, (idx+1) * dataloader.batch_size)
                
                x = model.forward_features(imgs.to(device))
                emb = pooling_embedding(x)
                out = model.forward_head(x)                                 
                
                emb = emb.data.cpu().numpy()
                batchprobs = F.softmax(out, dim=1).data.cpu().numpy() #Softmax 
                
                maxInds = np.argmax(batchprobs,1) # predicted label of model for each data 
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(emb[j]) * (1 - batchprobs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(emb[j]) * (-1 * batchprobs[j][c])
            return torch.Tensor(embedding)

    def query(self, model, n_subset: int = None) -> np.ndarray:
        gradEmbedding = self.get_grad_embedding(model, n_subset)
        chosen = init_centers(gradEmbedding, self.n_query)
        
        unlabeled_idx = np.where(self.labeled_idx==False)[0] 
        select_idx = unlabeled_idx[chosen]        
        return select_idx


@torch.no_grad()
def init_centers(X: torch.Tensor, K: int = None):
    
    # K-means ++ initializing 
    embs = torch.Tensor(X)
    ind = torch.argmax(torch.norm(embs, 2, 1)).item()
    embs = embs.cuda()
    
    mu = [embs[ind]]
    indsAll = [ind]
    centInds = [0.] * len(embs)
    cent = 0
    
    # Sampling 
    while len(mu) < K:
        if len(mu) == 1:
            D2 = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy()
        else:
            newD = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy()
            for i in range(len(embs)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]            
        D2 = D2.ravel().astype(float)
        
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        
        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


def pooling_embedding(x):
    '''
    dim : target dimension for pooling 
    mean : pooling for grad embedding 
    '''
    if np.argmax(list(x.shape)) == 1: # x shape : NHWC -> for Resnet 
        dim = (2,3)
        emb = x.mean(dim)
    elif np.argmax(list(x.shape)) == 3:# x shape : NCHW -> for Swin-Transformer 
        dim = (1,2)
        emb = x.mean(dim)
    elif len(x.shape) == 3: # x shape : NTC -> for ViT 
        emb = x[:,0,:] # cls token 
    return emb 
