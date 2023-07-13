import numbers
import numpy as np
import torch
import torch.nn.functional as F

import cv2 
from PIL import Image 
import torch.nn as nn 

from torchvision import transforms

def add_augmentation(transform: transforms.Compose, img_size: int, aug_info: list = None):
    augments_dict = {
        'RandomCrop': transforms.RandomCrop((img_size, img_size), padding=4),
        'RandomHorizontalFlip': transforms.RandomHorizontalFlip(),
        'RandomVerticalFlip': transforms.RandomVerticalFlip(),
        'Resize': transforms.Resize((img_size, img_size)),
        'PadWithKeepRatio': PadWithKeepRatio(padding_mode='reflect'),
        'CLAHE': CLAHE(),
        'EqualizeHist': EqualizeHist(),
    }
    # insert augmentations
    if aug_info != None:    
        for aug in aug_info:
            transform.transforms.insert(-1, augments_dict[aug])   
    else:
        transform.transforms.insert(-1, augments_dict['Resize'])
    
    return transform


def train_augmentation(img_size: int, mean: tuple, std: tuple, aug_info: list = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform = add_augmentation(transform=transform, img_size=img_size, aug_info=aug_info)

    return transform

def test_augmentation(img_size: int, mean: tuple, std: tuple):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])

    return transform


def get_padding(image):    
    # image shape: C x H x W
    h, w = image.shape[1:]
    # max size
    max_size = np.max([h, w])
    
    # height and width pad size
    h_pad = abs(max_size - h) / 2
    w_pad = abs(max_size - w) / 2
    
    # t: top, b: bottom, r: right, l: left
    t_pad = b_pad = h_pad if h_pad % 1 else h_pad + 0.5
    l_pad = r_pad = w_pad if w_pad % 1 else w_pad + 0.5
    
    # return padding
    padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))
    return padding

class PadWithKeepRatio(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img: torch.Tensor):
        """
        Args:
            img (torch.Tensor): Image to be padded.

        Returns:
            Padded image (torch.Tensor)
        """
        return F.pad(input=img, pad=get_padding(img), mode=self.padding_mode, value=self.fill)
    
    def __repr__(self):
        return self.__class__.__name__ + '(fill={0}, padding_mode={1})'.\
            format(self.fill, self.padding_mode)
            
            

class CLAHE(nn.Module):        
    def __init__(self, clipLimit: float = 2.0, tileGridSize: tuple = (8,8), p: float = 0.5):
        super(CLAHE,self).__init__()
        self.clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = tileGridSize)
        self.p = p
        
    def transform(self, img: np.ndarray or Image.Image):
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, torch.Tensor):
            raise TypeError            
        
        if len(img.shape) == 2 or img.shape[2] == 1:
                img = self.clahe.apply(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img[:, :, 0] = self.clahe.apply(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        
    def forward(self, img: np.ndarray or Image.Image):
        if np.random.random() < self.p:
            img = self.transform(img)
        
        return img

class EqualizeHist(nn.Module):
    def __init__(self, p: float = 0.5):
        super(EqualizeHist, self).__init__()
        self.equalize_hist = cv2.equalizeHist
        self.p = p
        
    def transform(self, img: np.ndarray or Image.Image):
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif type(img) == torch.Tensor:
            raise TypeError            
        
        if len(img.shape) == 2 or img.shape[2] == 1:
                img = self.equalize_hist(img)
        else:
            src_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
            ycrcb_planes = list(cv2.split(src_ycrcb))
            ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
            dst_ycrcb = cv2.merge(ycrcb_planes)
            img = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)
                
        return img
    
    def forward(self, img: np.ndarray or Image.Image):
        if np.random.random() < self.p:
            img = self.transform(img)
            
        return img