import omegaconf
from omegaconf import OmegaConf
from datasets import stats


def parser():
    args = OmegaConf.from_cli()
    # load default config
    cfg = OmegaConf.load(args.default_cfg)
    del args['default_cfg']
    
    if 'strategy_cfg' not in args.keys():
        del cfg['AL']
    
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
        
        # load resampler config
        if 'resampler_cfg' in args.keys():
            cfg_resampler = OmegaConf.load(args.resampler_cfg)
            cfg = OmegaConf.merge(cfg, cfg_resampler)
            del args['resampler_cfg']
        
    # merge config with new keys
    cfg = OmegaConf.merge(cfg, args)
    
    # Update experiment name
    cfg.DEFAULT.exp_name = cfg.AL.strategy if 'AL' in cfg.keys() else 'Full'
    if 'exp_name' in args.DEFAULT.keys():
        cfg.DEFAULT.exp_name = f'{cfg.DEFAULT.exp_name}-{args.DEFAULT.exp_name}'
       
    # load dataset statistics
    cfg.DATASET.update(stats.datasets[cfg.DATASET.name])
    
    if hasattr(cfg, 'AL'):
        # update stategy configs
        cfg = update_stategy_cfg(cfg)
        
        # update tta
        cfg = update_tta_crop_size(cfg)
        
        # change num_classes to nb_id_class for open-set AL
        if hasattr(cfg.AL, 'nb_id_class'):
            cfg.DATASET.num_classes = cfg.AL.nb_id_class
          
    print(OmegaConf.to_yaml(cfg))
    
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

def update_stategy_cfg(cfg):
    if 'PT4' in cfg.AL.strategy:
        cfg = _update_pt4al_cfg(cfg)
    
    return cfg


def update_openset_strategy_cfg(cfg, cfg_openset):
    if cfg_openset.AL.strategy in ['CLIPNAL']:
        cfg_openset.AL.openset_params.selected_strategy = cfg.AL.strategy
        
    return cfg_openset
        

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

def update_tta_crop_size(cfg):
    if hasattr(cfg.AL, 'tta_params'):
        for p in cfg.AL.tta_params:
            if isinstance(p, omegaconf.dictconfig.DictConfig):
                name = list(p)[0]
                if name == 'FiveCrop':
                    p[name].params.crop_size = (cfg.DATASET.img_size, cfg.DATASET.img_size)
                    
    return cfg