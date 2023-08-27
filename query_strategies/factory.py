import numpy as np
import torch
from torch.utils.data import Dataset

def create_query_strategy(
    model, strategy_name: str, dataset: Dataset, labeled_idx: np.ndarray, sampler_name: str,
    n_query: int, batch_size: int, num_workers: int, n_subset: int = 0, params: dict = dict()):
    
    strategy = __import__('query_strategies').__dict__[strategy_name](
        model        = model,
        n_query      = n_query, 
        n_subset     = n_subset,
        labeled_idx  = labeled_idx, 
        dataset      = dataset,
        sampler_name = sampler_name,
        batch_size   = batch_size,
        num_workers  = num_workers,
        **params
    )
    
    return strategy