from torchvision import transforms
from omegaconf.dictconfig import DictConfig

augments_dict = {
    'RandomCrop': transforms.RandomCrop,
    'RandomHorizontalFlip': transforms.RandomHorizontalFlip,
    'Resize': transforms.Resize
}

def add_augmentation(transform: transforms.Compose, img_size: int, aug_info: list = None):
    # insert augmentations
    if aug_info != None:    
        for aug in aug_info:
            print(aug)
            if isinstance(aug, dict) or isinstance(aug, DictConfig):
                for name, params in aug.items():
                    if name == 'RandomCrop':
                        transform.transforms.insert(-1, augments_dict[name]((img_size, img_size), **params))
                    else:
                        transform.transforms.insert(-1, augments_dict[name](**params))
            else:
                if aug == 'Resize':
                    transform.transforms.insert(-1, augments_dict[aug]((img_size, img_size)))
                else:
                    transform.transforms.insert(-1, augments_dict[aug]())   
    else:
        transform.transforms.insert(-1, augments_dict['Resize']((img_size, img_size)))
    
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