import numpy as np
import torch
from torch.utils.data import Dataset

def create_query_strategy(
    model, strategy_name: str, dataset: Dataset, labeled_idx: np.ndarray, 
    n_query: int, batch_size: int, num_workers: int, params: dict = dict()):
    
    strategy = __import__('query_strategies').__dict__[strategy_name](
        model       = model,
        n_query     = n_query, 
        labeled_idx = labeled_idx, 
        dataset     = dataset,
        batch_size  = batch_size,
        num_workers = num_workers,
        **params
    )
    
    return strategy