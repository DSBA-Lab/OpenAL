import numpy as np
from torch.utils.data import Dataset

def create_query_strategy(
        model, 
        strategy_name: str, 
        dataset: Dataset, 
        is_labeled: np.ndarray, 
        sampler_name: str, 
        transform,
        n_query: int, 
        batch_size: int, 
        num_workers: int, 
        steps_per_epoch: int = 0,
        n_subset: int = 0, 
        interval_type: str = 'top', 
        **params
    ):
    
    strategy = __import__('query_strategies').__dict__[strategy_name](
        model            = model,
        n_query          = n_query, 
        n_subset         = n_subset,
        is_labeled       = is_labeled, 
        dataset          = dataset,
        transform        = transform,
        sampler_name     = sampler_name,
        batch_size       = batch_size,
        num_workers      = num_workers,
        steps_per_epoch  = steps_per_epoch,
        interval_type    = interval_type,
        **params
    )
    
    return strategy