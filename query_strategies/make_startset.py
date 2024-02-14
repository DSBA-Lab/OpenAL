import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from .sampler import SubsetSequentialSampler
from .utils import get_target_from_dataset, torch_seed


def create_id_testloader(dataset, id_targets: np.ndarray, batch_size: int, num_workers: int):
    id_idx = [i for i in range(len(dataset.targets)) if dataset.targets[i] < len(id_targets)]
        
    sampler = SubsetSequentialSampler(indices=id_idx)
    
    dataloader = DataLoader(
        dataset     = dataset,
        sampler     = sampler,
        batch_size  = batch_size,
        num_workers = num_workers
    )

    return dataloader

def create_id_ood_targets(dataset, nb_id_class: int, seed: int):
    
    torch_seed(seed)
    id_targets = np.random.choice(a=range(len(dataset.classes)), size=nb_id_class, replace=False)
    target_map = dict(zip(id_targets, np.arange(len(id_targets))))
    
    new_targets = []
    targets = get_target_from_dataset(dataset=dataset)
    for t in targets:
        if t in target_map:
            new_targets.append(target_map[t])
        else:
            new_targets.append(len(target_map))
            
    dataset.targets = np.array(new_targets)
    
    # for ImageFolder class
    if dataset.__class__.__name__ == 'ImageNet':
        dataset.samples = [(s[0], t) for s, t in zip(dataset.samples, dataset.targets)]
    
    return dataset, id_targets


def create_is_labeled_unlabeled(trainset, id_targets: np.ndarray, size: int, ood_ratio: float, seed: int):
    '''
    Args:
    - trainset (torch.utils.data.Dataset): trainset
    - id_targets (np.ndarray): ID targets
    - size (int): represents the absolute number of train samples. 
    - ood_ratio (float): OOD class ratio
    - seed (int): seed for random state
    
    Return:
    - labeled_idx (np.ndarray): selected labeled indice
    - unlabeled_idx (np.ndarray): selected unlabeled indice
    '''
    
    torch_seed(seed)

    id_total_idx = [i for i in range(len(trainset.targets)) if trainset.targets[i] < len(id_targets)]
    ood_total_idx = [i for i in range(len(trainset.targets)) if trainset.targets[i] >= len(id_targets)]

    n_ood = round(len(id_total_idx) * (ood_ratio / (1 - ood_ratio)))
    ood_total_idx = random.sample(ood_total_idx, n_ood)
    print("# Total ID: {}, OOD: {}".format(len(id_total_idx), len(ood_total_idx)))

    _, lb_idx = train_test_split(id_total_idx, test_size=int(size * (1 - ood_ratio)), stratify=trainset.targets[id_total_idx], random_state=seed)
    ood_start_idx = random.sample(ood_total_idx, int(size * ood_ratio))
    ulb_idx = list(set(id_total_idx + ood_total_idx) - set(lb_idx) - set(ood_start_idx))
    print("# Labeled in: {}, ood: {}, Unlabeled: {}".format(len(lb_idx), len(ood_start_idx), len(ulb_idx)))

    # defined empty labeled index
    is_labeled = np.zeros(len(trainset), dtype=bool)
    is_unlabeled = np.zeros(len(trainset), dtype=bool)
    is_ood = np.zeros(len(trainset), dtype=bool)

    is_labeled[lb_idx] = True
    is_unlabeled[ulb_idx] = True
    is_ood[ood_start_idx] = True

    return is_labeled, is_unlabeled, is_ood


def create_is_labeled(method: str, trainset: Dataset, size: int, seed: int, **kwargs):
    '''
    Args:
    - method (str): 
    - trainset (torch.utils.data.Dataset): trainset
    - size (int): represents the absolute number of train samples. 
    - seed (int): seed for random state
    
    Return:
    - labeled_idx (np.ndarray): selected indice
    '''
    
    sample_idx = np.arange(len(trainset))
    
    if method == 'stratified_random_select':
        kwargs['stratify'] = get_target_from_dataset(dataset=trainset)
    
    # defined empty labeled index
    is_labeled = np.zeros_like(sample_idx, dtype=bool)
    
    # selected index
    selected_idx = eval(method)(sample_idx=sample_idx, size=size, seed=seed, **kwargs)
    
    is_labeled[selected_idx] = True
    
    return is_labeled


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
    b_size = int(b_size)

    if n_start > b_init: # if initial sample larger than batch size, then initial batch size is defined as initial sample
        b_init = n_start

    # sampling interval
    sampling_interval = int(b_init/n_start)
    
    return total_round, b_size, b_init, sampling_interval