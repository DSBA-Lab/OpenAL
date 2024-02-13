import os
import json
import numpy as np
import pandas as pd
import omegaconf
from subprocess import call
from copy import deepcopy
from torchvision import datasets

from .cifar_lt import CIFAR10LT, CIFAR100LT
from .waterbird import WaterBird
from .augmentation import create_augmentation

def load_cifar10(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):

    trainset = datasets.CIFAR10(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = True, 
        download  = True,
        transform = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )

    testset = datasets.CIFAR10(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = False, 
        download  = True,
        transform = create_augmentation(img_size=img_size, mean=mean, std=std)
    )
        
    return trainset, testset


def load_cifar100(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):

    trainset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = True, 
        download  = True,
        transform = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )

    testset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = False, 
        download  = True,
        transform = create_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


def load_cifar10lt(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None, 
                   imbalance_type: str = 'exp', imbalance_factor: int = 1):

    trainset = CIFAR10LT(
        root             = os.path.join(datadir,'CIFAR10'), 
        train            = True, 
        download         = True,
        transform        = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info),
        imbalance_type   = imbalance_type,
        imbalance_factor = imbalance_factor
    )

    testset = CIFAR10LT(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = False, 
        download  = True,
        transform = create_augmentation(img_size=img_size, mean=mean, std=std)
    )
        
    return trainset, testset


def load_cifar100lt(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None, 
                    imbalance_type: str = 'exp', imbalance_factor: int = 1):

    trainset = CIFAR100LT(
        root             = os.path.join(datadir,'CIFAR100'), 
        train            = True, 
        download         = True,
        transform        = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info),
        imbalance_type   = imbalance_type,
        imbalance_factor = imbalance_factor
    )

    testset = CIFAR100LT(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = False, 
        download  = True,
        transform = create_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


def load_svhn(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):

    trainset = datasets.SVHN(
        root      = os.path.join(datadir,'SVHN'), 
        split     = 'train', 
        download  = True,
        transform = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )

    testset = datasets.SVHN(
        root      = os.path.join(datadir,'SVHN'), 
        split     = 'test', 
        download  = True,
        transform = create_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


def load_tiny_imagenet_200(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):

    trainset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200','train'),
        transform = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )
    
    testset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200','val'),
        transform = create_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset

def load_imagenet1k(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):
    root = os.path.join(datadir,'imagenet1k')
    
    # download ImageNet1K
    def download_imagenet(r):
        os.makedirs(r, exist_ok=True)
        call(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --output-document={r}/ILSVRC2012_devkit_t12.tar.gz", shell=True)            
        call(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --output-document={r}/ILSVRC2012_img_val.tar", shell=True)            
        call(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --output-document={r}/ILSVRC2012_img_train.tar", shell=True)            
        
    if not os.path.exists(root):
        download_imagenet(root)

    # update aug_info for testset
    aug_info_test = ['CenterCrop']
    for aug in aug_info:
        if isinstance(aug, dict) or isinstance(aug, omegaconf.dictconfig.DictConfig):
            aug_name, aug_value = list(aug.items())[0]
            if aug_name == 'Resize':
                aug_info_test.insert(0, {aug_name: aug_value})

    trainset = datasets.ImageNet(
        root      = root,
        split     = 'train',
        transform = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )
    
    testset = datasets.ImageNet(
        root      = root,
        split     = 'val',
        transform = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info_test)
    )

    return trainset, testset
    

def load_waterbird(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):
    meta_info = pd.read_csv(os.path.join(datadir, 'metadata.csv'))

    trainset = WaterBird(
        datadir   = datadir,
        meta_info = meta_info[meta_info.split == 0],
        transform = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )
    
    validset = WaterBird(
        datadir   = datadir,
        meta_info = meta_info[meta_info.split == 1],
        transform = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )
    
    testset = WaterBird(
        datadir   = datadir,
        meta_info = meta_info[meta_info.split == 2],
        transform = create_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )

    return trainset, validset, testset


def create_dataset(
    datadir: str, dataname: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None, **params
):

    datasets = eval(f"load_{dataname.lower()}")(
        datadir  = datadir, 
        img_size = img_size,
        mean     = mean, 
        std      = std,
        aug_info = aug_info,
        **params
    )

    if dataname != 'WaterBird':
        # benchmark datasets
        trainset, testset = datasets
        validset = deepcopy(testset)
        
    elif dataname == 'WaterBird':
        trainset, validset, testset = datasets
    
    # generate imbalance data after splitting datasets
    # because CIFAR-LT assumes that test-set is a balanced dataset
    if dataname == 'CIFAR10LT' or dataname == 'CIFAR100LT':
        trainset.gen_imbalanced_data()
        
    # set classes
    trainset = set_classes(dataset=trainset, dataname=dataname)
    validset = set_classes(dataset=validset, dataname=dataname)
    testset = set_classes(dataset=testset, dataname=dataname)
    
    # set dataname
    trainset.dataname = dataname.lower()
    validset.dataname = dataname.lower()
    testset.dataname = dataname.lower()
    
    # set img size
    trainset.img_size = (img_size, img_size)
    validset.img_size = (img_size, img_size)
    testset.img_size = (img_size, img_size)
    
    # set mean and std
    trainset.stats = {'mean':mean, 'std':std}
    validset.stats = {'mean':mean, 'std':std}
    testset.stats = {'mean':mean, 'std':std}
    
    return trainset, validset, testset


def set_classes(dataset, dataname):
    is_need_update = ['Tiny_ImageNet_200', 'ImageNet1K']
    if not hasattr(dataset, 'classes') or dataname in is_need_update:
        print('add classes from classnames.json')
        classnames = json.load(open('classnames.json', 'r'))[dataname.lower()]
        dataset.classes = np.array(classnames)
        
    if isinstance(dataset.classes, list):
        dataset.classes = np.array(dataset.classes)
        
    return dataset
    