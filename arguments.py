from omegaconf import OmegaConf
import argparse
from datasets import stats

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
    cfg_default = OmegaConf.load(args.default_setting)
    
    # merge strategy config
    cfg_strategy = OmegaConf.load(args.strategy_setting)
    cfg = OmegaConf.merge(cfg_default, cfg_strategy)
    
    # Update experiment name
    cfg.DEFAULT.exp_name = cfg.AL.strategy if 'AL' in cfg.keys() else 'Full'
    
    # update cfg
    for k, v in zip(args.opts[0::2], args.opts[1::2]):
        if k == 'DEFAULT.exp_name':
            cfg.DEFAULT.exp_name = f'{cfg.DEFAULT.exp_name}-{v}'
        else:
            try:
                OmegaConf.update(cfg, k, eval(v), merge=True)
            except:
                OmegaConf.update(cfg, k, v, merge=True)
       
    # load dataset statistics
    cfg.DATASET.update(stats.datasets[cfg.DATASET.dataname])
    
    print(OmegaConf.to_yaml(cfg))
    
    return cfg  