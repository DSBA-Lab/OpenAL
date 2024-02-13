import omegaconf
from torchvision import transforms

def create_augmentation(img_size: int, mean: tuple, std: tuple, transform: list = None, aug_info: list = None, **kwargs):
    # base transform
    if transform == None:
        transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
            
    # insert augmentations
    if aug_info != None:    
        # update image size
        aug_info = update_img_size(img_size=img_size, aug_info=aug_info)
        
        # create augmentation
        for aug in aug_info:
            if isinstance(aug, str):
                transform.insert(-1, __import__('torchvision.transforms', fromlist='transforms').__dict__[aug]())
            elif isinstance(aug, dict) or isinstance(aug, omegaconf.dictconfig.DictConfig):
                aug_name, aug_value = list(aug.items())[0]
                transform.insert(-1, __import__('torchvision.transforms', fromlist='transforms').__dict__[aug_name](**aug_value))
    
    return transforms.Compose(transform)


def update_img_size(img_size: int, aug_info: list):
    
    is_need_size = ['RandomCrop', 'CenterCrop', 'Resize', 'RandomResizedCrop']
    
    for i, aug in enumerate(aug_info):
        if isinstance(aug, str):
            if aug in is_need_size:
                aug_info[i] = {aug: {'size': img_size}}
        elif isinstance(aug, dict) or isinstance(aug, omegaconf.dictconfig.DictConfig):
            aug_name, aug_value = list(aug.items())[0]
            if aug_name in is_need_size and 'size' not in aug_value:
                aug_value['size'] = img_size
                aug_info[i] = {aug_name: aug_value}
        
    return aug_info
