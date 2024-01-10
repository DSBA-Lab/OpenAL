import torch
import random
import os
import numpy as np

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


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)