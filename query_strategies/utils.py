import torch
import random
import os
import numpy as np
import re
import json
from PIL import Image
from _ctypes import PyObj_FromPtr  # see https://stackoverflow.com/a/15012814/355230
from torch.utils.data import IterableDataset

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
    
    
class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('Only lists and tuples can be wrapped')
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                    else super(MyEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded
            
            
class TrainIterableDataset(IterableDataset):
    def __init__(self, dataset, sample_idx: np.ndarray = [], return_index: bool = False):
        for k, v in dataset.__dict__.items():
            setattr(self, k, v)
        
        if len(sample_idx) == 0:
            self.sample_idx = np.arange(len(dataset))
        else:
            self.sample_idx = sample_idx if len(sample_idx) < len(dataset) else np.arange(len(dataset))
            
        self.return_index = return_index

    def generate(self):
        while True:
            idx = np.random.choice(a=self.sample_idx, size=1, replace=False)[0]
            img, target = self.data[idx], self.targets[idx]
            
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            img = self.transform(img)
        
            if self.return_index:
                yield idx, img, target
            else:
                yield img, target

    def __iter__(self):
        return iter(self.generate())
    
    
class IndexDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        for k, v in dataset.__dict__.items():
            setattr(self, k, v)
            
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return idx, img, target
    
    def __len__(self):
        return len(self.dataset)