from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset

import os
import pandas as pd


def create_labeled_index(method: str, trainset: Dataset, size: int, seed: int, **kwargs):
    '''
    Args:
    - method (str): 
    - sample_idx (list): data indice
    - size (int): represents the absolute number of train samples. 
    - seed (int): seed for random state
    
    Return:
    - labeled_idx (list): selected indice
    '''
    
    sample_idx = np.arange(len(trainset))
    
    if method == 'stratified_random_select':
        kwargs['stratify'] = get_target_from_dataset(dataset=trainset)
    
    # defined empty labeled index
    labeled_idx = np.zeros_like(sample_idx, dtype=bool)
    
    # selected index
    selected_idx = eval(method)(sample_idx=sample_idx, size=size, seed=seed, **kwargs)
    labeled_idx[selected_idx] = True
    
    return labeled_idx

def get_target_from_dataset(dataset):
    # if class name is ALDataset
    if dataset.__class__.__name__ == "ALDataset":
        targets = dataset.data_info.label.tolist()
    else:
       # attribution name list in benchmark dataset class
        target_attrs = ['targets', 'labels'] # TODO: if target attribution name is added, append in this line.

        # iterativly check attribution name if not False else break
        for attr in target_attrs:
            targets = getattr(dataset, attr, False)
            if targets is not False:
                break

    return targets


def random_select(sample_idx: list, size: int, seed: int):
    '''
    Args:
    - sample_idx (list): data indice
    - size (int): represents the absolute number of train samples. 
    - seed (int): seed for random state
    
    Return:
    - select_idx (list): selected indice
    '''
    
    np.random.seed(seed)
    np.random.shuffle(sample_idx)
    selected_idx = sample_idx[:size]
    
    return selected_idx
    
    

def stratified_random_select(sample_idx: list, size: int, seed: int, stratify: list):
    '''
    Args:
    - sample_idx (list): data indice
    - size (int): represents the absolute number of train samples. 
    - seed (int): seed for random state
    - stratify (list): data is split in a stratified fashion, using this as the class labels.
    
    Return:
    - select_idx (list): selected indice
    '''
    select_idx, _ = train_test_split(
        sample_idx, 
        train_size   = size, 
        stratify     = stratify, 
        random_state = seed
    )
    
    return select_idx
    
    
def batch_select(sample_idx: list, size: int, **kwargs):
    '''
    Select intital sample using batch index of SSL results.
    ex) PT4AL uses batch index sorted by SSL(rotation) loss in descending order
    
    Args:
    - size (int): represents the absolute number of train samples. 
    - batch_path (str): pt4al batch loss path.
    
    Return:
    - selecte_idx (list): selected indice.
    
    
    If 'b_init' is not divided by 'sampling_interval' by 'size', select_idx is sliced using 'size' because more indexes are extracted.
    
    ex1)
    len(batch_idx): 50,000
    len(size): 1,000
    n_end: 10,000
    n_query: 1,000
    
    print(total_round)
    > 10
    
    print(b_size)
    > 5000
    
    print(b_init)
    > 5000
    
    print(sampling_interval)
    > 5
    
    print(len(range(0, b_init, sampling_interval)))
    > 1000
    
    
    ex2)
    len(batch_idx): 34,000
    len(size): 1,000
    n_end: 10,000
    n_query: 1,000
    
    print(total_round)
    > 10
    
    print(b_size)
    > 3400
    
    print(b_init)
    > 3400
    
    print(sampling_interval)
    > 3
    
    print(len(range(0, b_init, sampling_interval)))
    > 1134
    
    '''
    
    # load ssl pretext batch
    batch_idx = pd.read_csv(kwargs['batch_path'])['idx'].values
    assert len(batch_idx) == len(sample_idx), 'sample_idx and batch_idx must same length.'
    
    # get params for batch sampling
    _, _, b_init, sampling_interval = get_batch_params(
        batch_size = len(batch_idx),
        n_start    = size,
        n_end      = kwargs['n_end'],
        n_query    = kwargs['n_query']
    )

    ## first bach uniform sampling
    selected_idx = batch_idx[range(0, b_init, sampling_interval)][:size]
    
    return selected_idx


def get_batch_params(batch_size: int, n_start: int, n_end: int, n_query: int):
    # total round that includes first round
    total_round = ((n_end - n_start)/n_query) + 1
    if total_round % 2 != 0:
        total_round = int(total_round) + 1
    else:
        total_round = int(total_round)

    # batch size
    b_size = batch_size/total_round
    assert int(b_size) > n_query, 'the number of query must smaller than batch size.'
    
    # initial batch size
    b_init = int(b_size) if b_size % 2 == 0 else int(b_size) + 1
        
    if batch_size > b_init: # if initial sample larger than batch size, then initial batch size is defined as initial sample
        b_init = batch_size

    # sampling interval
    sampling_interval = int(b_init/batch_size)
    
    return total_round, b_size, b_init, sampling_interval