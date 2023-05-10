# Active Learning 

삼성 S.LSI - Active Learning 프로젝트 (2023)

# Environments

NVIDIA pytorch docker [ [link](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12) ]

```bash
docker pull nvcr.io/nvidia/pytorch:22.12-py3
```

`requirements.txt`

```bash
accelerate
wandb
torchvision
```


# Methods

`./query_strategies`

- Random Sampling (baseline)
- Least Confidence
- Margin Sampling
- Entropy Sampling
- Learning Loss for Active Leanring



# TODO

- [x] Subset Module for Unlabeled Pool
- Learning Loss
    - [x] Learning Loss
    - [ ] Optimize seperate model weights of backbone and Loss Prediction Module
- [ ] Feature Mixing
- [ ] Survey setting
    - [x] Learning loss (w/o seperate)
    - [ ] entropy sampling
- [ ] PT4AL
- [ ] k-center greedy
- [ ] BALD

