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
    targets = trainset.data_info.label.tolist()
    
    if method == 'stratified_random_select':
        kwargs['stratify'] = targets
    
    # defined empty labeled index
    labeled_idx = np.zeros_like(sample_idx, dtype=bool)
    
    # selected index
    selected_idx = eval(method)(sample_idx=sample_idx, size=size, seed=seed, **kwargs)
    labeled_idx[selected_idx] = True
    
    return labeled_idx


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
    
    
def pt4al_select(size: int, **kwargs):
    '''
    Args:
    - size (int): represents the absolute number of train samples. 
    - save_path (str): pt4al batch loss path.
    - sampling_interval (int): uniform sampling interval.
    
    Return:
    - selecte_idx (list): selected indice.
    '''
    
    ## load ssl pretext batch
    ssl_batch_path = os.path.join(kwargs['save_path'], 'batch', 'batch_loss.txt')
    
    with open(ssl_batch_path, 'r') as f:
        samples = f.readlines()
    
    ## first bach uniform sampling
    samples = np.array(samples)
    if 'CIFAR10' in kwargs['save_path']:
        selecte_idx = samples[[j*kwargs['sampling_interval'] for j in range(size)]]
    elif 'SamsungAL' in kwargs['save_path']:
        selecte_idx = samples[:size]
    selecte_idx = list(map(int, pd.DataFrame(selecte_idx)[0].str.replace('\n', '')))
    
    return selecte_idx