import numpy as np
from scipy import stats
from copy import deepcopy as deepcopy
import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from .strategy import Strategy
from .sampler import SubsetSequentialSampler

class BADGE(Strategy):
    def __init__(self, **init_args):
        
        super(BADGE, self).__init__(**init_args)            
        
    def get_grad_embedding(self, model, unlabeled_idx: np.ndarray) -> torch.Tensor:
        # define sampler
        sampler = SubsetSequentialSampler(indices=unlabeled_idx)
        
        # unlabeled dataloader
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers
        )
        
        # Prepare to get gradient embedding 
        embDim = model.num_features # number of features of embedding 
        device = next(model.parameters()).device 
        nLab = model.num_classes # the number of classes
        embedding = np.zeros([len(sampler), embDim * nLab]) # Empty Tensor 
        
        model.eval()
        with torch.no_grad():
            for idx, (imgs,y) in enumerate(dataloader):
                idxs = np.arange(idx* dataloader.batch_size, (idx+1) * dataloader.batch_size)
                
                x = model.forward_features(imgs.to(device))
                emb = pooling_embedding(x) # used to calculate gradient embedding 
                out = model.forward_head(x) 
                
                emb = emb.data.cpu().numpy()
                batchprobs = F.softmax(out, dim=1).data.cpu().numpy() # Softmax 
                                
                # Calculate Gradient Embedding as uncertainty
                maxInds = np.argmax(batchprobs,1) # predicted label of model for each data 
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(emb[j]) * (1 - batchprobs[j][c])  # gradient embedding : differentiation of CrossEntropy = (p-I(y=i)) * z(x;V); p=1, z(x;V)=emb
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(emb[j]) * (-1 * batchprobs[j][c]) # gradient embedding : differentiation of CrossEntropy = (p-I(y=i)) * z(x;V); p=0, z(x;V)=emb
            return torch.Tensor(embedding)

    def query(self, model) -> np.ndarray:
        # unlabeled index
        unlabeled_idx = self.get_unlabeled_idx()
        
        # get gradients of embedding
        gradEmbedding = self.get_grad_embedding(model=model, unlabeled_idx=unlabeled_idx)
        chosen = init_centers(X=gradEmbedding, K=self.n_query)
        
        select_idx = unlabeled_idx[chosen]        
        return select_idx


@torch.no_grad()
def init_centers(X: torch.Tensor, K: int = None):    
    '''
    K-MEANS++ seeding algorithm 
    - 초기 센터는 gradient embedding의 norm이 가장 큰 것으로 사용 
    - 그 이후 부터는 앞서 선택 된 center와의 거리를 계산, 이와 비례하는 확률로 이산 확률 분포를 만듬
    - 이 이산 확률 분포에서 새로운 센터를 선택 
    - 이렇게 뽑힌 센터들은 k-means 목적함수의 기대값에 근사되는 것이 증명 됨, 따라서 다양성을 확보할 수 있음 (Arthur and Vassilvitskii, 2007)
    '''    
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
            D2 = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy() # Calculate l2 Distance btw mu and embs 
        else:
            newD = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy() # Calculate l2 Distance btw mu and embs 
            for i in range(len(embs)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]            
        D2 = D2.ravel().astype(float)
        
        Ddist = (D2 ** 2)/ sum(D2 ** 2) 
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist)) # 이산 확률 분포 구축 
        ind = customDist.rvs(size=1)[0] # 이산 확률 분포에서 하나 추출 
        
        while ind in indsAll: ind = customDist.rvs(size=1)[0] # repeat until choosing index not in indsAll 
        
        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


def pooling_embedding(x):
    '''
    dim : Target dimension for pooling 
    mean : Average Pooling 
    '''
    if np.argmax(list(x.shape)) == 1:   # x shape : NCHW  -> for Resnet 
        dim = (2,3)
        emb = x.mean(dim)
    elif np.argmax(list(x.shape)) == 3: # x shape : NHWC -> for Swin-Transformer 
        dim = (1,2)
        emb = x.mean(dim)
    elif len(x.shape) == 3: # x shape : NTC -> for ViT 
        emb = x[:,0,:]      # cls token 
    return emb 
