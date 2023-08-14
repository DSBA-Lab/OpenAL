from omegaconf import OmegaConf
import argparse
from datasets import stats

def convert_type(value):
    # None
    if value == 'None':
        return None
    
    # bool
    for t in [bool, float, int, list, dict]:
        check, value = str_to_type(value=value, type=t)
        if check:
            return value
    
    return value

def str_to_type(value, type):
    try:
        check = isinstance(eval(value), type)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
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
    
    # update stategy configs
    cfg = update_stategy_cfg(cfg)
    
    print(OmegaConf.to_yaml(cfg))
    
    return cfg  



def parser_ssl():
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



def update_stategy_cfg(cfg):
    if 'PT4' in cfg.AL.strategy:
        cfg = _update_pt4al_cfg(cfg)
        
    return cfg


def _update_pt4al_cfg(cfg):
    # for AL params
    cfg.AL.params.n_start = cfg.AL.n_start
    cfg.AL.params.n_end = cfg.AL.n_end
    
    # for AL init params
    cfg.AL.init.params = {
        'batch_path' : cfg.AL.params.batch_path,
        'n_query'    : cfg.AL.n_query,
        'n_end'      : cfg.AL.n_end
    }

    return cfg        