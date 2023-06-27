from omegaconf import OmegaConf
import argparse
from datasets import stats

def convert_type(value):
    print(value)
    print(type(value))
    # None
    if value == 'None':
        return None
    
    # list or tuple
    elif len(value.split(',')) > 1:
        return value.split(',')
    
    # bool
    check, value = str_to_bool(value)
    if check:
        return value
    
    # float
    check, value = str_to_float(value)
    if check:
        return value
    
    # int
    check, value = str_to_int(value)
    if check:
        return value
    
    return value

def str_to_bool(value):
    try:
        if isinstance(eval(value), bool):
            return True, eval(value)
        else:
            return False, value
    except NameError:
        return False, value
    
def str_to_float(value):
    try:
        if isinstance(float(value), float):
            return True, float(value)
        else:
            False, value
    except ValueError:
        return False, value
    
def str_to_int(value):
    try:
        check = isinstance(int(value), int)
        return True, int(value) if check else False, value
    except ValueError:
        return False, value

def parser():
    parser = argparse.ArgumentParser(description='Active Learning - Benchmark')
    parser.add_argument('--default_setting', type=str, default=None, help='default config file')
    parser.add_argument('--strategy_setting', type=str, default=None, help='strategy config file')    
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load default config
    cfg = OmegaConf.load(args.default_setting)
    
    # load strategy config
    if args.strategy_setting:
        cfg_strategy = OmegaConf.load(args.strategy_setting)
        cfg = OmegaConf.merge(cfg, cfg_strategy)
    else:
        del cfg['AL']
    
    # Update experiment name
    cfg.DEFAULT.exp_name = cfg.AL.strategy if 'AL' in cfg.keys() else 'Full'
    
    # update cfg
    for k, v in zip(args.opts[0::2], args.opts[1::2]):
        if k == 'DEFAULT.exp_name':
            cfg.DEFAULT.exp_name = f'{cfg.DEFAULT.exp_name}-{v}'
        else:
            OmegaConf.update(cfg, k, convert_type(v), merge=True)
       
    # load dataset statistics
    cfg.DATASET.update(stats.datasets[cfg.DATASET.dataname])
    
    print(OmegaConf.to_yaml(cfg))
    
    return cfg  