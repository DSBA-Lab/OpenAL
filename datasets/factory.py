import os
from .build import ALDataset
from .augmentation import train_augmentation, test_augmentation
    
    
def create_dataset_benchmark(datadir: str, dataname: str, img_size: int, mean: tuple, std: tuple):
    trainset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
        root      = os.path.join(datadir, dataname), 
        train     = True, 
        download  = True, 
        transform = train_augmentation(img_size=img_size, mean=mean, std=std)
    )
    testset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
        root      = os.path.join(datadir, dataname), 
        train     = False, 
        download  = True, 
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


def create_dataset(datadir: str, dataname: str, img_size: int, mean: tuple, std: tuple):
    trainset = ALDataset(
        datadir   = os.path.join(datadir,dataname),
        name      = 'train.csv',
        transform = train_augmentation(img_size=img_size, mean=mean, std=std)
    )
    validset = ALDataset(
        datadir   = os.path.join(datadir,dataname), 
        name      = 'validation.csv',
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )
    testset = ALDataset(
        datadir   = os.path.join(datadir,dataname), 
        name      = 'test.csv',
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )
    
    return trainset, validset, testset    