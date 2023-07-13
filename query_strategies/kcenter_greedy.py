import numpy as np
import torch
from torch.utils.data import Dataset
import cvxpy as cp

from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int):
        
        super(EntropySampling, self).__init__(
            model       = model,
            n_query     = n_query, 
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
    def query(self, model, n_subset: int = None) -> np.ndarray:
        
        # predict probability on unlabeled dataset
        probs = self.extract_unlabeled_prob(model = model, n_subset = n_subset)
        
        # unlabeled index
        unlabeled_idx = np.where(self.labeled_idx == False)[0]
        # Y = np.where(self.labeled_idx == True)[0]
        self.dataset
        
        # select maximum entropy
        entropy = (-(probs * torch.log(probs))).sum(dim = 1)

        # if 'default' in model :
        # select_idx = unlabeled_idx[entropy.sort(descending = True)[1][:self.n_query]]
	    
        # elif 'CBAL' in  :
        b = self.n_query
        N = len(unlabeled_idx)
        # L1_DISTANCE = []
        # L1_Loss = []
        # ENT_Loss = []
        probs = probs.numpy()
        entropy = entropy.numpy()

        if self.dataset == 'cifar10':
            num_classes = 10
        elif self.dataset == 'cifar100':
            num_classes = 100
        elif self.dataset == 'SamsungAL':
            num_classes = 4

        labeled_classes = Y[self.labeled_idx]
        _, counts = np.unique(labeled_classes, return_counts=True)

        class_threshold = int(( 2 * self.n_query + (self.cycle + 1) * self.n_query )/ num_classes)
        class_share = class_threshold - counts
        samples_share= np.array([ 0 if c<0 else c for c in class_share ]).reshape(num_classes,1)

        if self.dataset == 'cifar10':
            lamda = 0.6
        elif self.dataset == 'cifar100':
            lamda = 2
        elif self.dataset == 'SamsungAL':
            lamda = 0.3

        for lam in [lamda]:
            z = cp.Variable((N, 1),boolean=True)
            constraints = [sum(z) == b]
            cost = z.T @ entropy + lam * cp.norm1(probs.T @ z - samples_share)
            objective = cp.Minimize(cost)
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.GUROBI, verbose=True, TimeLimit=1000)

            print('Optimal value with gurobi : ', problem.value)
            print(problem.status)
            print("A solution z is")
            print(z.value.T)
            
            lb_flag = np.array(z.value.reshape(1, N)[0], dtype=bool)
				
			# 	# -----------------Stats of optimization---------------------------------
			# 	ENT_Loss.append(np.matmul(z.value.T, U))
			# 	print('ENT LOSS= ', ENT_Loss)
			# 	threshold = (2 * self.n_query / num_classes) + (self.cycle + 1) * self.n_query / num_classes
			# 	round=self.cycle+1
			# 	freq = torch.histc(torch.FloatTensor(self.Y[unlabeled_idx[lb_flag]]), bins=num_classes)+torch.histc(torch.FloatTensor(self.Y[self.idxs_lb]), bins=num_classes)
			# 	L1_distance = (sum(abs(freq - threshold)) * num_classes / (2 * (2 * self.n_query + round * self.n_query) * (num_classes - 1))).item()
			# 	print('Lambda = ',lam)
			# 	L1_DISTANCE.append(L1_distance)
			# 	L1_Loss_term=np.linalg.norm(np.matmul( probs.T, z.value ) - samples_share, ord=1)
			# 	L1_Loss.append(L1_Loss_term)

			# print('L1 Loss = ')
			# for i in L1_Loss:
			# 	print('%.3f' %i)
			# print('L1_distance = ')
			# for j in L1_DISTANCE:
			# 	print('%.3f' % j)
			# print('ENT LOSS = ')
			# for k in ENT_Loss:
			# 	print('%.3f' % k)
				
            select_idx = unlabeled_idx[lb_flag]
	
        return select_idx