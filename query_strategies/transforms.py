import torch
from typing import List
from itertools import product
from torchvision import transforms
from torchvision.transforms import functional as F

class TransformParams:
    def __init__(self, params: dict):
        self.params = params
        self.product_params()
        
    def product_params(self):
        params_set = []
        for p in list(product(*self.params.values())):
            params_set.append(dict(zip(self.params.keys(), p)))
            
        setattr(self, 'params_set', params_set)
                
        
class FiveResizedCrop(TransformParams):
    def __init__(self):
        self.params = {'func_name': ['crop_lt','crop_lb','crop_rt','crop_rb','crop_center']}
        super(FiveResizedCrop, self).__init__(params = self.params)        
        
    def __call__(self, img: torch.Tensor, func_name: str):
        # img: (B x C x H x W)
        _, _, h, w = img.size()
        h_half = h//2
        w_half = w//2
            
        img = F.resize(img=img, size=(h+h_half, w+w_half))
    
        return self.__getattribute__(func_name)(img, crop_size=(h, w))
    
    def crop_lt(self, img: torch.Tensor, crop_size: tuple):
        """crop left top corner"""
        return img[:, :, 0:crop_size[0], 0:crop_size[1]]


    def crop_lb(self, img: torch.Tensor, crop_size: tuple):
        """crop left bottom corner"""
        return img[:, :, -crop_size[0]:, 0:crop_size[1]]


    def crop_rt(self, img: torch.Tensor, crop_size: tuple):
        """crop right top corner"""
        return img[:, :, 0:crop_size[0], -crop_size[1]:]


    def crop_rb(self, img: torch.Tensor, crop_size: tuple):
        """crop right bottom corner"""
        return img[:, :, -crop_size[0]:, -crop_size[1]:]


    def crop_center(self, img: torch.Tensor, crop_size: tuple):
        """make center crop"""

        img = F.center_crop(img, output_size=crop_size)
        return img
    
    
class Rotation(TransformParams):
    def __init__(
        self, angle: List[float], interpolation: List[str] = [transforms.InterpolationMode.NEAREST], 
        fill: List[float] = [None]):
        
        self.params = {
            'angle'         : set([0] + angle),
            'interpolation' : interpolation,
            'fill'          : fill
        }
        
        super(Rotation, self).__init__(params = self.params) 
        
        
    def __call__(self, img, angle, interpolation, fill):
        img = F.rotate(img=img, angle=angle, interpolation=interpolation, fill=fill)
        
        return img
    
class HorizontalFlip(TransformParams):
    def __init__(self):
        self.params = {'apply': [False, True]}
        super(HorizontalFlip, self).__init__(params = self.params) 
        
    def __call__(self, img, apply):
        if apply:
            img = F.hflip(img=img)
        return img