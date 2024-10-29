import os
import json
from accelerate import Accelerator
from omegaconf import OmegaConf

import timm
from timm import create_model

import torch

from arguments import parser
from datasets import create_dataset
from datasets.utils import get_features
from main import make_directory

def run(cfg):
    savedir = os.path.join(
        cfg.DEFAULT.savedir, cfg.DATASET.name, cfg.MODEL.name
    )
    make_directory(savedir=savedir)
    
    # save config
    OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
    
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )
    
    trainset, _, testset = create_dataset(
        datadir  = cfg.DATASET.datadir, 
        dataname = cfg.DATASET.name,
        img_size = cfg.DATASET.img_size,
        mean     = cfg.DATASET.mean,
        std      = cfg.DATASET.std,
        aug_info = cfg.DATASET.aug_info,
        **cfg.DATASET.get('params', {})
    )

    model = create_model(cfg.MODEL.name, pretrained=True, img_size=cfg.DATASET.img_size, num_classes=0)
    data_config = timm.data.resolve_model_data_config(model)
    data_config['input_size'] = (3, cfg.DATASET.img_size, cfg.DATASET.img_size)
    trainset.transform = timm.data.create_transform(**data_config, is_training=False)
    testset.transform = timm.data.create_transform(**data_config, is_training=False)

    train_features = get_features(
        dataset     = trainset,
        model       = model,
        batch_size  = cfg.DATASET.batch_size,
        num_workers = cfg.DATASET.num_workers,
        device      = accelerator.device
    )
    test_features = get_features(
        dataset     = testset,
        model       = model,
        batch_size  = cfg.DATASET.batch_size,
        num_workers = cfg.DATASET.num_workers,
        device      = accelerator.device
    )
    
    torch.save(train_features, os.path.join(savedir, 'train_features.pt'))
    torch.save(test_features, os.path.join(savedir, 'test_features.pt'))
    
if __name__ == '__main__':
    # config
    cfg = parser()

    # run
    run(cfg)