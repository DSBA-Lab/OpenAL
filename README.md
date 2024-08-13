# OpenAL

**OpenAL** is an Active Learning (AL) framework for classification.

The AL is classified into two approaches by samples of unlabeled data. The AL that assumes the unlabeled data contains only in-distribution data is called Standard AL. If the unlabeled data includes not only in-distribution but also out-of-distribution, it is called Open-set AL. This framework covers Standard AL and Open-set AL. So, we named our framework **OpenAL**.

We hope that AL research can advance further through this framework.

# Environments

We build environments based on docker image `nvcr.io/nvidia/pytorch:22.12-py3`.

```
python==3.8.10
torch==1.14.0a0+410ce96
torchvision==0.15.0a0
accelerate==0.18.0
wandb
torchvision==0.15.0a0
omegaconf
timm==0.9.2
seaborn==0.12.2
torchlars==0.1.2
ftfy==6.1.3
open-clip-torch==2.24.0
finch-clust==0.1.9
```

# Query Strategies

AL | Type | Method | `Class Name` | Paper
---|---|---|---|---
None-AL | - | Random Sampling | `RandomSampling` | -
Standard AL | Uncertainty | Least Confidence | `LeastConfidence` | IJCNN 2014 - [paper](https://ieeexplore.ieee.org/document/6889457)
Standard AL | Uncertainty | Margin Sampling | `MarginSampling` | IJCNN 2014 - [paper](https://ieeexplore.ieee.org/document/6889457)
Standard AL | Uncertainty | Entropy | `EntropySampling` | IJCNN 2014 - [paper](https://ieeexplore.ieee.org/document/6889457)
Standard AL | Uncertainty | VarRatio | `VarRatio` | ICMLW 2020 - [paper](https://arxiv.org/abs/2007.06364)
Standard AL | Uncertainty | MeanSTD | `MeanSTD` | CVPRW 2016 - [paper](https://ieeexplore.ieee.org/document/7789580/authors#authors)
Standard AL | Uncertainty | Learning Loss | `LearningLossAL` | CVPR 2019 - [paper](https://arxiv.org/abs/1905.03677), [unofficial](https://github.com/Mephisto405/Learning-Loss-for-Active-Learning)
Standard AL | Uncertainty | AlphaMix | `AlphaMixSampling` | CVPR 2022 - [paper](https://arxiv.org/abs/2203.07034), [official](https://github.com/AminParvaneh/alpha_mix_active_learning)
Standard AL | Uncertainty | BALD | `BALD` | arXiv 2011.12 - [paper](https://arxiv.org/abs/1112.5745), [unofficial](https://github.com/lunayht/DBALwithImgData)
Standard AL | Hybrid | BADGE | `BADGE` | NeurIPS 2019 - [paper](https://arxiv.org/abs/1906.08158), [official](https://github.com/JordanAsh/badge)
Standard AL | Diversity | K-Center Greedy | `KCenterGreedy` | CVPR 2018 - [paper](https://arxiv.org/abs/1708.00489), [official](https://github.com/ozansener/active_learning_coreset)
Standard AL | Diversity | K-Center Greedy + Class Balanced | `KCenterGreedyCB` | WACV 2022 - [paper](https://arxiv.org/abs/2110.04543), [official](https://github.com/Javadzb/Class-Balanced-AL)
Open-set AL | Contrastive Learning | CCAL | `CCAL` | ICCV 2021 - [paper](https://openaccess.thecvf.com//content/ICCV2021/html/Du_Contrastive_Coding_for_Active_Learning_Under_Class_Distribution_Mismatch_ICCV_2021_paper.html), [official](https://github.com/RUC-DWBI-ML/CCAL)
Open-set AL | Contrastive Learning | MQNet | `MQNet` | NeurIPS 2022 - [paper](https://arxiv.org/abs/2210.07805), [official](https://github.com/kaist-dmlab/MQNet/tree/main)
Open-set AL | OOD Detector | LfOSA | `LfOSA` | CVPR 2022 - [paper](https://arxiv.org/abs/2201.06758), [official](https://github.com/ningkp/LfOSA/tree/master)
Open-set AL | OOD Detector | EOAL | `EOAL` | AAAI 2024 - [paper](https://arxiv.org/abs/2312.14126),  [official](https://github.com/bardisafa/EOAL)
Open-set AL | VLM | CLIPNAL | `CLIPNAL` | arXiv 2024.8 - [paper](https://arxiv.org/abs/2408.04917), [official](https://github.com/DSBA-Lab/OpenAL)

## CLIPN checkpoint

`CLIPNAL` uses a CLIPN checkpoint shared from [CLIPN repository](https://github.com/xmed-lab/CLIPN/tree/main?tab=readme-ov-file). The checkpoint can download in [here](https://drive.google.com/drive/folders/1qF4Pm1JSL3P0H4losPSmvldubFj91dew?usp=sharing).

# Configuration for Experiments

All configuration files is in `./configs`.  
You can modify the config files to run your experiment setttings.

```
./configs
├── default_setting.yaml
├── openset_al
│   ├── ccal.yaml
│   ├── clipnal.yaml
│   ├── eoal.yaml
│   ├── lfosa.yaml
│   └── mqnet.yaml
├── ssl
│   ├── csi.yaml
│   └── simclr.yaml
└── standard_al
    ├── badge.yaml
    ├── bald.yaml
    ├── entropy_sampling.yaml
    ├── featmix_sampling.yaml
    ├── kcenter_greedy_cb.yaml
    ├── kcenter_greedy.yaml
    ├── learning_loss.yaml
    ├── least_confidence.yaml
    ├── margin_sampling.yaml
    ├── meanstd_sampling.yaml
    ├── random_sampling.yaml
    └── varratio_sampling.yaml
```


# How to Use

## Standard AL

```python
from query_strategies import create_query_strategy

model = # classifier
trainset = # training data set with labeled and unlabeled samples
transform = # transforms for extracting features
sampler_name = # SubsetRandomSampler or SubsetWeightedRandomSampler
is_labeled = # bool type 1-d array (N,). N is the number of samples. True is labeled samples and False is unlabeled samples. 
n_query = # the number of samples for annotation
n_subset = # sampling size for unlabeled data
batch_size = # batch size
num_workers = # number of workers

strategy = create_query_strategy(
    strategy_name    = # strategy name, 
    model            = model,
    dataset          = trainset, 
    transform        = transform,
    sampler_name     = sampler_name,
    is_labeled       = is_labeled, 
    n_query          = n_query, 
    n_subset         = n_subset,
    batch_size       = batch_size, 
    num_workers      = num_workers
)

# select query using trained model on labeled samples
query_idx = strategy.query(model)
strategy.update(query_idx=query_idx)
```

## Open-set AL

```python
from query_strategies import create_query_strategy

model = # classifier
trainset = # training data set with labeled and unlabeled samples
transform = # transforms for extracting features
sampler_name = # SubsetRandomSampler or SubsetWeightedRandomSampler
is_labeled = # bool type 1-d array (N,). N is the number of samples. True is labeled ID samples and False is unlabeled samples. 
n_query = # the number of samples for annotation
n_subset = # sampling size for unlabeled data
batch_size = # batch size
num_workers = # number of workers

# select strategy    
openset_params = {
    'is_openset'      : # if unlabeled data contains OOD samples, True, or False
    'is_unlabeled'    : # bool type 1-d array (N,). N is the number of samples. True is unlabeled samples and False is labeled ID and OOD samples.
    'is_ood'          : # bool type 1-d array (N,). N is the number of samples. True is OOD samples and False is unlabeled and ID samples.
    'id_classes'      : # ID class names
    'savedir'         : # save directory
    'seed'            : # seed
}

strategy = create_query_strategy(
    strategy_name    = # strategy name, 
    model            = model,
    dataset          = trainset, 
    transform        = transform,
    sampler_name     = sampler_name,
    is_labeled       = is_labeled, 
    n_query          = n_query, 
    n_subset         = n_subset,
    batch_size       = batch_size, 
    num_workers      = num_workers,
    **openset_params
)

# select query using trained model on labeled samples
query_idx = strategy.query(model)
id_query_idx = strategy.update(query_idx=query_idx)
```

# How to Run

**Supervised Learning with full train dataset**
```bash
python main.py \
default_cfg=./configs/default_setting.yaml \
DATASET.name=$dataname \
DEFAULT.savedir=$savedir
```

**Standard AL**
```bash
python main.py \
default_cfg=./configs/default_setting.yaml \
strategy_cfg=./configs/standard_al/$strategy_name.yaml \
DATASET.name=$dataname \
AL.n_start=$n_start \
AL.n_query=$n_query \
AL.n_end=$n_end \
DEFAULT.savedir=$savedir
```

**Open-set AL**
```bash
python main.py \
default_cfg=./configs/default_setting.yaml \
openset_cfg=./configs/openset_al/$strategy_name.yaml \
DATASET.name=$dataname \
AL.ood_ratio=$ood_ratio \
AL.id_ratio=$id_ratio \
AL.n_start=$n_start \
AL.n_query=$n_query \
AL.n_end=$n_end \
DEFAULT.savedir=$savedir
```
