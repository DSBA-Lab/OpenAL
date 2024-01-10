import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

def get_target_from_dataset(dataset):
    # if class name is ALDataset
    if dataset.__class__.__name__ == "ALDataset":
        targets = dataset.data_info.label.values
    else:
       # attribution name list in benchmark dataset class
        target_attrs = ['targets', 'labels'] # TODO: if target attribution name is added, append in this line.

        # iterativly check attribution name if not False else break
        for attr in target_attrs:
            targets = getattr(dataset, attr, False)
            if targets is not False:
                break

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    return targets


def insert_selected_samples(dataset, selected_idx: list):
    # ImageFolder class
    if dataset.__class__.__name__ == 'ImageFolder':
        selected_samples = np.array(dataset.samples)[selected_idx]
        selected_samples = [(p, int(c)) for p, c in selected_samples]
        dataset.samples = selected_samples 
        
        return dataset
    
    
    # select samples
    if hasattr(dataset, 'data'):
        dataset.data = dataset.data[selected_idx]
    
    # select targets   
    # attribution name list in benchmark dataset class
    target_attrs = ['targets', 'labels'] # TODO: if target attribution name is added, append in this line.

    # iterativly check attribution name if not False else break
    for attr in target_attrs:
        if hasattr(dataset, attr):
            dataset.__dict__[attr] = np.array(dataset.__dict__[attr])[selected_idx]
            
    return dataset


def split_data(trainset, seed: int):
    
    test_ratio = 0.1 # this is a fixed ratio
    validset = deepcopy(trainset)
    
    # stratified split datasets
    train_idx, valid_idx = train_test_split(
        range(len(trainset)),
        test_size    = int(len(trainset) * test_ratio),
        stratify     = get_target_from_dataset(dataset=trainset),
        random_state = seed
    )
    
    # insert selected samples into dataset
    trainset = insert_selected_samples(dataset=trainset, selected_idx=train_idx)
    validset = insert_selected_samples(dataset=validset, selected_idx=valid_idx)
    
    return trainset, validset