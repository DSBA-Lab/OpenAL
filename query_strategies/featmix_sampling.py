import copy
import math

import numpy as np
from torch.autograd import Variable

from .strategy import Strategy, SubsetSequentialSampler
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class AlphaMixSampling(Strategy):
	def __init__(self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int):
		super(AlphaMixSampling, self).__init__(
			model       = model,
            n_query     = n_query, 
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers)
		self.device = next(model.parameters()).device

	def query(self, model, n_subset: int = None) -> np.ndarray:
		
		n = self.n_query
		unlabeled_idx = np.where(self.labeled_idx==False)[0]

		ulb_probs, org_ulb_embedding = self.extract_unlabeled_prob_embed(model=model, n_subset=n_subset)

		_, probs_sort_idxs = ulb_probs.sort(descending=True)
		pred_1 = probs_sort_idxs[:, 0]

		_, org_lb_embedding, Y = self.extract_labeled_prob_embed(model=model, n_subset=n_subset)

		ulb_embedding = org_ulb_embedding
		lb_embedding = org_lb_embedding

		unlabeled_size = ulb_embedding.size(0)
		embedding_size = ulb_embedding.size(1)

		min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
		candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

		var_emb = Variable(ulb_embedding, requires_grad=True).to(self.device)
		out = self.model.fc_forward(var_emb)
		loss = F.cross_entropy(out, pred_1.to(self.device))
		grads = torch.autograd.grad(loss, var_emb)[0].data.cpu()
		del loss, var_emb, out
		
		alpha_cap = 0.
		while alpha_cap < 1.0:
			alpha_cap += 0.03125

			tmp_pred_change, tmp_min_alphas = \
				self.find_candidate_set(
					lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap=alpha_cap,
					Y=Y,
					grads=grads)

			is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

			min_alphas[is_changed] = tmp_min_alphas[is_changed]
			candidate += tmp_pred_change

			if candidate.sum() > n:
				break

		if candidate.sum() > 0:
			print('Number of inconsistencies: %d' % (int(candidate.sum().item())))

			print('alpha_mean_mean: %f' % min_alphas[candidate].mean(dim=1).mean().item())
			print('alpha_std_mean: %f' % min_alphas[candidate].mean(dim=1).std().item())
			print('alpha_mean_std %f' % min_alphas[candidate].std(dim=1).mean().item())

			c_alpha = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()

			selected_idxs = self.sample(min(n, candidate.sum().item()), feats=c_alpha)
			selected_idxs = unlabeled_idx[candidate][selected_idxs]
		else:
			selected_idxs = np.array([], dtype=np.int)

		if len(selected_idxs) < n:
			remained = n - len(selected_idxs)
			idx_lb = copy.deepcopy(self.idxs_lb)
			idx_lb[selected_idxs] = True
			selected_idxs = np.concatenate([selected_idxs, np.random.choice(np.where(idx_lb == 0)[0], remained)])
			print('picked %d samples from RandomSampling.' % (remained))

		return np.array(selected_idxs)

	def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, Y, grads):

		unlabeled_size = ulb_embedding.size(0)
		embedding_size = ulb_embedding.size(1)

		min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
		pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

		alpha_cap /= math.sqrt(embedding_size)
		grads = grads.to(self.device)
		
		n_label = torch.max(Y)+1
		print(n_label)
		for i in range(n_label):
			emb = lb_embedding[Y.view(-1) == i]
			if emb.size(0) == 0:
				emb = lb_embedding
			anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

			embed_i, ulb_embed = anchor_i.to(self.device), ulb_embedding.to(self.device)
			alpha = self.calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)

			embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i
			out = self.model.fc_forward(embedding_mix)
			out = out.detach().cpu()
			alpha = alpha.cpu()

			pc = out.argmax(dim=1) != pred_1
			
			torch.cuda.empty_cache()

			alpha[~pc] = 1.
			pred_change[pc] = True
			is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
			min_alphas[is_min] = alpha[is_min]
			
		return pred_change, min_alphas

	def sample(self, n, feats):
		feats = feats.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(feats)

		cluster_idxs = cluster_learner.predict(feats)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (feats - centers) ** 2
		dis = dis.sum(axis=1)
		return np.array(
			[np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
			 (cluster_idxs == i).sum() > 0])
	
	def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
		z = (lb_embedding - ulb_embedding)
		alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)

		return alpha


	def extract_unlabeled_prob_embed(self, model, n_subset: int = None) -> torch.Tensor:         
        
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

		# predict
		probs = []
		embeds = []

		device = next(model.parameters()).device
		model.eval()
		with torch.no_grad():
			for i, (inputs, _) in enumerate(dataloader):
				outputs, embed = model.embed_forward(inputs.to(device))
				outputs = torch.nn.functional.softmax(outputs, dim=1)
				probs.append(outputs.cpu())
				embeds.append(embed.cpu())

		return torch.vstack(probs), torch.vstack(embeds)
    
	def extract_labeled_prob_embed(self, model, n_subset: int = None) -> torch.Tensor:         
        
		# define sampler
		labeled_idx = np.where(self.labeled_idx==True)[0]
		sampler = SubsetSequentialSampler(
			indices = self.subset_sampling(indices=labeled_idx, n_subset=n_subset) if n_subset else labeled_idx
		)

		# unlabeled dataloader
		dataloader = DataLoader(
			dataset     = self.dataset,
			batch_size  = self.batch_size,
			sampler     = sampler,
			num_workers = self.num_workers
		)

		# predict
		probs = []
		embeds = []
		Y = []

		device = next(model.parameters()).device
		model.eval()
		with torch.no_grad():
			for i, (inputs, label) in enumerate(dataloader):
				outputs, embed = model.embed_forward(inputs.to(device))
				outputs = torch.nn.functional.softmax(outputs, dim=1)
				probs.append(outputs.cpu())
				embeds.append(embed.cpu())
				Y.append(label.view(label.size(0),1).cpu())

		return torch.vstack(probs), torch.vstack(embeds), torch.vstack(Y)