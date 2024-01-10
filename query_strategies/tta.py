import torch
import omegaconf
from itertools import product
from typing import List

class TTA:
    def __init__(self, agg: str, params: dict):
        assert agg in ['mean','max','min'], 'agg should be one of mean, max, and min'
        
        self.tta_params = params
        self.agg = agg
        self.build_transforms()
        self.tta_params_set = list(product(*[p.params_set for p in self.tta_transforms]))
    
    def build_transforms(self):
        tta_transforms = []
        for t in self.tta_params:
            if isinstance(t, str):
                tta_transforms.append(__import__('query_strategies.transforms', fromlist='transforms').__dict__[t]())
            elif isinstance(t, dict) or isinstance(t, omegaconf.dictconfig.DictConfig):
                name = list(t)[0]
                tta_transforms.append(__import__('query_strategies.transforms', fromlist='transforms').__dict__[name](**t[name].get('params', {})))
        
        setattr(self, 'tta_transforms', tta_transforms)
        
    def apply(self, img: torch.Tensor, params: list) -> torch.Tensor:
        for t, p in zip(self.tta_transforms, params):
            img = t(img, **p)

        return img
    
    def aggregate(self, outputs: list) -> torch.Tensor:
        # outputs: (nb augs x B x C x H x W)
        outputs = torch.stack(outputs)
        
        # aggregation
        if self.agg == 'mean':
            outputs = outputs.mean(dim=0)
        elif self.agg == 'max':
            outputs, _ = outputs.max(dim=0)
            
        return outputs

    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        # img: (B x C x H x W)
        out = []
        for params in self.tta_params_set:
            img_i = self.apply(img=img, params=params)
            out.append(img_i)

        return out