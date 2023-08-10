import numpy as np
import torch
from torch.utils.data import Dataset

from .strategy import Strategy

# VarRatioSampling : Sampling by variation and ratio
# variation과 ratio를 이용하여 샘플링 진행

class VarRatioSampling(Strategy):
    def __init__(
        self, model, n_query: int, labeled_idx: np.ndarray, 
        dataset: Dataset, batch_size: int, num_workers: int):
        
        super(VarRatioSampling, self).__init__(
            model       = model,
            n_query     = n_query, 
            labeled_idx = labeled_idx, 
            dataset     = dataset,
            batch_size  = batch_size,
            num_workers = num_workers
        )
        
    def query(self, model, n_subset: int = None) -> np.ndarray:
        # Variation Ratios Sampling
        # Reference : Elementary applied statistics : for students in dehavioral science. New York: Wiley, 1965

        # 라벨링되지 않은 데이터셋에 대해 확률값 예측
        probs = self.extract_unlabeled_prob(model = model, n_subset = n_subset)
        
        _preds = probs.max(1)[0]
        _uncertainties = (1.0 - _preds).sort(descending = True)[1
        ]

        # unlabeled index
        unlabeled_idx = np.where(self.labeled_idx==False)[0]

        select_idx = unlabeled_idx[_uncertainties[:self.n_query]]

        return select_idx