from _ctypes import PyObj_FromPtr  # see https://stackoverflow.com/a/15012814/355230
import json
import re
import os
import pandas as pd
from omegaconf import OmegaConf

from glob import glob

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
            
            
def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, [v]))
    return dict(items)
            
            
def save_full_results(path: str, savedir: str) -> pd.DataFrame:
    """
    Args:
    - path (str): results directory
    - savedir (str): results to save directory
    
    Returns:
    - df (pd.DataFrame): full images supervised learning results
    """
    os.makedirs(savedir, exist_ok=True)
    
    cfg_path = glob(os.path.join(path, '*/Full*/configs.yaml'))

    df = pd.DataFrame()
    for p in cfg_path:
        # results directory
        r_dir = os.path.dirname(p)
        best_path = os.path.join(r_dir, 'results_seed0_best.json')
        test_path = os.path.join(r_dir, 'results-seed0.json')
        
        # unfinished
        if not os.path.isfile(test_path):
            continue
        
        # configs
        cfg = OmegaConf.load(p)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = flatten_dict(cfg)
        
        # valid best results
        best_r = json.load(open(best_path, 'r'))
        cfg['best_step'] = [best_r.pop('best_step')]
        for k, v in best_r.items():
            cfg[f'best_{k}'] = [v]
        
        # test results
        test_r = json.load(open(test_path,'r'))
        for k, v in test_r.items():
            cfg[f'test_{k}'] = [v]
        
        # add results directory
        cfg['dir'] = r_dir
        
        df = pd.concat([df, pd.DataFrame(cfg)], axis=0)
        
    df.to_csv(os.path.join(savedir, 'full_resutls.csv'), index=False)