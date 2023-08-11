import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from omegaconf import OmegaConf
import argparse
from datasets import stats
from arguments import *

def parser():
    parser = argparse.ArgumentParser(description='Active Learning - SSL')
    parser.add_argument('--ssl_setting', type=str, default=None, help='SSL config file')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load default config
    cfg = OmegaConf.load(args.ssl_setting)
    
    # assert experiment name
    assert cfg.DEFAULT.get('exp_name', False) != False, 'exp_name is not defined.'
    
    # update cfg
    for k, v in zip(args.opts[0::2], args.opts[1::2]):
        OmegaConf.update(cfg, k, convert_type(v), merge=True)
       
    # load dataset statistics
    cfg.DATASET.update(stats.datasets[cfg.DATASET.dataname])
    
    print(OmegaConf.to_yaml(cfg))
    
    return cfg  