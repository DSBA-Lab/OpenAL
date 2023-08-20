# Active Learning 

삼성 S.LSI - Active Learning 프로젝트 (2023)

# Environments

NVIDIA pytorch docker [ [link](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12) ]

```bash
docker pull nvcr.io/nvidia/pytorch:22.12-py3
```

`requirements.txt`

```
accelerate==0.18.0
wandb
torchvision==0.15.0a0
gradio==3.27.0
omegaconf
timm==0.9.2
seaborn==0.12.2
```


# Methods

`./query_strategies`

1. Random Sampling (baseline)
2. Least Confidence
3. Margin Sampling
4. Entropy Sampling
5. Learning Loss for Active Leanring (+detach)
6. BALD
7. Badge
8. MeanSTD
9. VarRatio
10. Feature Mixing
11. PT4AL
