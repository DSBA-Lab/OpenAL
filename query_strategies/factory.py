import numpy as np
from torch.utils.data import Dataset

def create_query_strategy(
        model, 
        strategy_name: str, 
        dataset: Dataset, 
        is_labeled: np.ndarray, 
        sampler_name: str, 
        test_transform,
        n_query: int, 
        batch_size: int, 
        num_workers: int, 
        steps_per_epoch: int = 0,
        n_subset: int = 0, 
        tta_agg: str = None, 
        tta_params: dict = None, 
        interval_type: str = 'top', 
        resampler_params: dict = None,
        **params
    ):
    
    strategy = __import__('query_strategies').__dict__[strategy_name](
        model            = model,
        n_query          = n_query, 
        n_subset         = n_subset,
        is_labeled       = is_labeled, 
        dataset          = dataset,
        test_transform   = test_transform,
        sampler_name     = sampler_name,
        batch_size       = batch_size,
        num_workers      = num_workers,
        steps_per_epoch  = steps_per_epoch,
        tta_agg          = tta_agg,
        tta_params       = tta_params,
        interval_type    = interval_type,
        resampler_params = resampler_params,
        **params
    )
    
    return strategy