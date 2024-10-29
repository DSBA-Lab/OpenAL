import omegaconf
from omegaconf import OmegaConf
from datasets import stats


def parser(args: omegaconf.dictconfig.DictConfig = None, print_args: bool = True):
    if args == None:
        args = OmegaConf.from_cli()
        
    # load default config
    cfg = OmegaConf.load(args.default_cfg)
    del args['default_cfg']
    
    # load strategy config
    if 'strategy_cfg' in args.keys():
        cfg_strategy = OmegaConf.load(args.strategy_cfg)
        cfg = OmegaConf.merge(cfg, cfg_strategy)
        del args['strategy_cfg']
    
    # load openset config
    if 'openset_cfg' in args.keys():           
        cfg_openset = OmegaConf.load(args.openset_cfg)
        cfg_openset = update_openset_strategy_cfg(cfg, cfg_openset)
        cfg = OmegaConf.merge(cfg, cfg_openset)
        
        del args['openset_cfg']
        
    if cfg.get('AL'):
        if not cfg.AL.get('strategy'):
            del cfg['AL']
        
    # merge config with new keys
    cfg = update_cfg(cfg, args)
    
    # Update experiment name
    if 'AL' in cfg.keys():
        if cfg.AL.strategy in ['CLIPNAL', 'MQNet']:
            cfg.DEFAULT.exp_name = f'{cfg.AL.strategy}-{cfg.AL.openset_params.selected_strategy}'
        else:
            cfg.DEFAULT.exp_name = f'{cfg.AL.strategy}'
    else:
        cfg.DEFAULT.exp_name = 'Full'
    
    if 'exp_name' in args.DEFAULT.keys():
        cfg.DEFAULT.exp_name = f'{cfg.DEFAULT.exp_name}-{args.DEFAULT.exp_name}'
       
    # load dataset statistics
    cfg.DATASET.update(stats.datasets[cfg.DATASET.name])
    
    if cfg.DATASET.get('use_predefined_id_targets', False):
        cfg.DATASET.predefined_id_targets = stats.predefined_id_targets[cfg.DATASET.name]
        cfg.AL.nb_id_class = len(cfg.DATASET.predefined_id_targets)
        cfg.DATASET.num_classes = cfg.AL.nb_id_class
    else:
        if hasattr(cfg, 'AL'):        
            # change num_classes to nb_id_class for open-set AL
            if hasattr(cfg.AL, 'id_ratio'):
                cfg.AL.nb_id_class = int(cfg.DATASET.num_classes*cfg.AL.id_ratio)
                cfg.DATASET.num_classes = cfg.AL.nb_id_class
        
        # change num_classes to nb_id_class for full supervised learning
        if hasattr(cfg.DATASET, 'id_ratio'):
            cfg.DATASET.nb_id_class = int(cfg.DATASET.num_classes*cfg.DATASET.id_ratio)
            cfg.DATASET.num_classes = cfg.DATASET.nb_id_class
        
    if print_args:
        print(OmegaConf.to_yaml(cfg))
    
    return cfg  

def update_cfg(cfg, args):
    if args.get('OPTIMIZER', False):
        if args.OPTIMIZER.get('name'):
            if (args.OPTIMIZER.name != cfg.OPTIMIZER.name) and args.OPTIMIZER.get('params', False):
                cfg.OPTIMIZER.params = args.OPTIMIZER.params
                
    if args.get('SCHEDULER', False):
        if args.SCHEDULER.get('name'):
            if (args.SCHEDULER.name != cfg.SCHEDULER.name) and args.SCHEDULER.get('params', False):
                cfg.SCHEDULER.params = args.SCHEDULER.params
                
    cfg = OmegaConf.merge(cfg, args)
    
    return cfg

def parser_ssl():
    args = OmegaConf.from_cli()

    # load default config
    cfg = OmegaConf.load(args.ssl_cfg)
    
    # merge config with new keys
    cfg = OmegaConf.merge(cfg, args)
    
    # assert experiment name
    assert cfg.DEFAULT.get('exp_name', False) != False, 'exp_name is not defined.'
       
    # load dataset statistics
    cfg.DATASET.update(stats.datasets[cfg.DATASET.dataname])
    
    print(OmegaConf.to_yaml(cfg))
    
    return cfg

def update_openset_strategy_cfg(cfg, cfg_openset):
    if cfg_openset.AL.strategy in ['CLIPNAL', 'MQNet']:
        cfg_openset.AL.openset_params.selected_strategy = cfg.AL.strategy
        
    return cfg_openset
        