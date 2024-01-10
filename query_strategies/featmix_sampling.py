import copy
import math
import logging
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .strategy import Strategy

_logger = logging.getLogger('train')

class AlphaMixSampling(Strategy):
    def __init__(self, alpha_cap: float = 0.03125, **init_args):
        super(AlphaMixSampling, self).__init__(**init_args)
        
        self.alpha_cap = alpha_cap
        
        # the number of samples is selected by random sampling because the number of candidates is smalleer then the number of query
        self.remain = 0

    def query(self, model, **kwargs) -> np.ndarray:
        device = next(model.parameters()).device
        
        # unlabeled index
        unlabeled_idx = kwargs.get('unlabeled_idx', self.get_unlabeled_idx())
            
        # predict probability and embedding on unlabeled dataset
        ulb_outputs = self.extract_outputs(
            model        = model, 
            sample_idx   = unlabeled_idx, 
            return_embed = True
        )
        ulb_probs, ulb_embed = ulb_outputs['probs'], ulb_outputs['embed']

        # top-1 predicted class
        _, ulb_pred_sort_idx = ulb_probs.sort(dim=1, descending=True)
        ulb_pred_1 = ulb_pred_sort_idx[:, 0]

        # predict probability and embedding on labeled dataset
        lb_outputs = self.extract_outputs(
            model         = model, 
            sample_idx    = np.where(self.is_labeled==True)[0],
            return_probs  = False,
            return_embed  = True,
            return_labels = True
        )
        
        lb_embed, labels = lb_outputs['embed'], lb_outputs['labels']


        with torch.no_grad():
            with torch.enable_grad():
                # calculate gradients of unlabeled embedding with respect to loss using ulb_embed_out andulb_pred_1
                ulb_embed = Variable(ulb_embed, requires_grad=True)
                ulb_embed_out = model.forward_head(ulb_embed.to(device)).cpu()
                ulb_loss = F.cross_entropy(ulb_embed_out, ulb_pred_1)
                ulb_grads = torch.autograd.grad(ulb_loss, ulb_embed)[0]
                
            ulb_embed = ulb_embed.detach().cpu()
            del ulb_loss, ulb_embed_out
            
            # init min_alphas and candidate
            nb_ulb, ulb_emb_size = ulb_embed.size(0), np.prod(ulb_embed.size()[1:])
            candidate = torch.zeros(nb_ulb, dtype=torch.bool)
            min_alphas = torch.ones((nb_ulb, ulb_emb_size), dtype=torch.float)
            
            # epsilon is a hyper-parameter governing the magnitude of the mixing.
            # Intuitively, this optimisation chooses the hardest case of alpha(α) for each unlabelled instance and anchor
            epsilon = 0.
            
            # check candidate
            while epsilon < 1.0:
                epsilon += self.alpha_cap

                pred_change_i, min_alphas_i = self.find_candidate_set(
                    model      = model,
                    epsilon    = epsilon,
                    lb_embed   = lb_embed, 
                    labels     = labels,
                    ulb_embed  = ulb_embed, 
                    ulb_pred_1 = ulb_pred_1,
                    ulb_grads  = ulb_grads
                )

                is_changed = min_alphas.norm(dim=1) >= min_alphas_i.norm(dim=1)

                min_alphas[is_changed] = min_alphas_i[is_changed]
                candidate += pred_change_i

                _logger.info("The number of selected candidates: [{}/{}], epsilon: {}".format(candidate.sum(), len(candidate), epsilon))
                if candidate.sum() >= self.n_query:
                    break

            # if the number of candidates is the same as the number of query, then return selected idx
            if candidate.sum() == self.n_query: 
                selected_idx = unlabeled_idx[candidate]
                return selected_idx
            
            # if the number of candidates is not the same as the number of query, select query using K-Means
            if candidate.sum() > 0: 
                c_alpha = F.normalize(
                    F.adaptive_avg_pool2d(ulb_embed[candidate], 1).flatten(start_dim=1), 
                    p   = 2, 
                    dim = 1
                ).detach()

                # sampling query using K-Mean
                selected_idx = self.sample(n=min(self.n_query, candidate.sum().item()), feats=c_alpha)
                selected_idx = unlabeled_idx[candidate][selected_idx]
            
            else:
                selected_idx = np.array([], dtype=np.int)

            # if the number of candiates is smaller then the number of query, select query using random sampling from unlabeled index
            if len(selected_idx) < self.n_query:
                self.remained = self.n_query - len(selected_idx)
                
                # copy labeled index to sample an insufficient number from unlabeled index
                is_labeled = copy.deepcopy(self.is_labeled)
                is_labeled[selected_idx] = True
                selected_idx = np.concatenate([selected_idx, np.random.choice(np.where(self.is_labeled == False)[0], self.remained)])

        setattr(self, 'min_alphas', min_alphas)

        return selected_idx

    def find_candidate_set(
        self, model, epsilon: float, lb_embed: torch.Tensor, labels: torch.Tensor, 
        ulb_embed: torch.Tensor, ulb_pred_1: torch.Tensor, ulb_grads: torch.Tensor):

        device = next(model.parameters()).device

        # init min_alphas and pred_change
        nb_ulb, ulb_emb_size = ulb_embed.size(0), np.prod(ulb_embed.size()[1:])
        pred_change = torch.zeros(nb_ulb, dtype=torch.bool)
        min_alphas = torch.ones((nb_ulb, ulb_emb_size), dtype=torch.float)
        
        # devided by unlabeled embedding size
        epsilon /= math.sqrt(ulb_emb_size)
        
        # find prediction change per class
        num_classes = len(torch.unique(labels))
        for c_i in range(num_classes):
            lb_embed_c_i = lb_embed[labels == c_i]
            
            # TODO: there is no embedding of class i-th labeled data?
            if lb_embed_c_i.size(0) == 0:
                lb_embed_c_i = lb_embed
            
            # anchor_c_i is an average embedding of class i-th labeled data
            anchor_c_i = lb_embed_c_i.mean(dim=0).view(1, -1).repeat(nb_ulb, 1)
            
            # get optimized alpha
            alpha = self.calculate_optimum_alpha(
                eps       = epsilon, 
                lb_embed  = anchor_c_i, 
                ulb_embed = ulb_embed.flatten(start_dim=1),
                ulb_grads = ulb_grads.flatten(start_dim=1)
            )

            # feature mixing
            embedding_mix = (1 - alpha) * ulb_embed.flatten(start_dim=1) + alpha * anchor_c_i
            
            # inference using feature mixing
            out = model.forward_head(
                embedding_mix.view(ulb_embed.size()).to(device)
            ).cpu()

            # prediction change
            pc = out.argmax(dim=1) != ulb_pred_1

            # log changed samples
            alpha[~pc] = 1.
            pred_change[pc] = True
            is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
            min_alphas[is_min] = alpha[is_min]
            
        return pred_change, min_alphas

    def calculate_optimum_alpha(self, eps: float, lb_embed: torch.Tensor, ulb_embed: torch.Tensor, ulb_grads: torch.Tensor):
        z = (lb_embed - ulb_embed)
        alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)

        return alpha

    def sample(self, n: int, feats: torch.Tensor):
        feats = feats.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(feats)

        # prediction cluster
        cluster_idxs = cluster_learner.predict(feats)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = ((feats - centers) ** 2).sum(axis=1)
        
        return np.array([
            np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) 
            if (cluster_idxs == i).sum() > 0
        ])

