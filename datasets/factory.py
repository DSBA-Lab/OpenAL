import os

from torchvision import datasets
from .build import ALDataset
from .cifar_lt import CIFAR10LT, CIFAR100LT
from .augmentation import train_augmentation, test_augmentation

def load_cifar10(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):

    trainset = datasets.CIFAR10(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = True, 
        download  = True,
        transform = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )

    testset = datasets.CIFAR10(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = False, 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )
        
    return trainset, testset


def load_cifar100(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):

    trainset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = True, 
        download  = True,
        transform = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )

    testset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = False, 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


def load_cifar10lt(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None, 
                   imbalance_type: str = 'exp', imbalance_factor: int = 200):

    trainset = CIFAR10LT(
        root             = os.path.join(datadir,'CIFAR10'), 
        train            = True, 
        download         = True,
        transform        = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info),
        imbalance_type   = imbalance_type,
        imbalance_factor = imbalance_factor
    )

    testset = CIFAR10LT(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = False, 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )
        
    return trainset, testset


def load_cifar100lt(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None, 
                    imbalance_type: str = 'exp', imbalance_factor: int = 200):

    trainset = CIFAR100LT(
        root             = os.path.join(datadir,'CIFAR100'), 
        train            = True, 
        download         = True,
        transform        = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info),
        imbalance_type   = imbalance_type,
        imbalance_factor = imbalance_factor
    )

    testset = CIFAR100LT(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = False, 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


def load_svhn(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):

    trainset = datasets.SVHN(
        root      = os.path.join(datadir,'SVHN'), 
        split     = 'train', 
        download  = True,
        transform = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )

    testset = datasets.SVHN(
        root      = os.path.join(datadir,'SVHN'), 
        split     = 'test', 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


def load_tiny_imagenet_200(datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None):

    trainset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200','train'),
        transform = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )
    
    testset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200','val'),
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset
    
def load_al_dataset(dataname: str, datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None, seed: int = 42):
    trainset = ALDataset(
        datadir   = os.path.join(datadir, dataname),
        name      = f'train_seed{seed}.csv',
        transform = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info)
    )
    validset = ALDataset(
        datadir   = os.path.join(datadir, dataname), 
        name      = f'validation_seed{seed}.csv',
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )
    testset = ALDataset(
        datadir   = os.path.join(datadir, dataname), 
        name      = f'test_seed{seed}.csv',
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )
    
    return trainset, validset, testset


def create_dataset(
    datadir: str, dataname: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None, **params
):
   
    if f"load_{dataname.lower()}" in __import__(__name__).__dict__.keys():
        # benchmark datasets
        trainset, testset = eval(f"load_{dataname.lower()}")(
            datadir  = datadir, 
            img_size = img_size,
            mean     = mean, 
            std      = std,
            aug_info = aug_info,
            **params
        )
        
        validset = testset
        
    else:
        # user custom datasets
        trainset, validset, testset = load_al_dataset(
            datadir  = datadir, 
            dataname = dataname,
            img_size = img_size,
            mean     = mean,
            std      = std,
            aug_info = aug_info,
            **params
        )
    
    return trainset, validset, testset