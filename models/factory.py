from timm import create_model as _create_model
import torch.nn as nn

def create_model(
    modelname: str, num_classes: int = 0, pretrained: bool = False, img_size: int = 32, **params):
    
    query_model = None
    
    if modelname == 'DualPromptAL':
        query_model = _create_model(
            params['encoder_name'], 
            num_classes = num_classes,
            pretrained  = True, 
            img_size    = params['img_size'], 
            patch_size  = params['patch_size']
        )
        query_model.eval()
        
        model = __import__('models').__dict__[modelname](
            num_classes = num_classes,
            pretrained  = pretrained,
            **params
        )
    elif modelname == 'VPTAL':
        model = __import__('models').__dict__[modelname](
            num_classes = num_classes,
            pretrained  = pretrained,
            **params
        )
    else:
        model = _create_model(
            modelname, 
            num_classes = num_classes,
            pretrained  = pretrained, 
            **params
        )
    
        if not pretrained and img_size < 224:
            if 'conv1' in model._modules.keys():
                model.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=3, padding=1, stride=1, bias=False)
                model.maxpool = nn.Identity()
            elif 'stem' in model._modules.keys():
                model.stem.conv = nn.Conv2d(3, model.stem.conv.out_channels, kernel_size=3, padding=1, stride=1, bias=False)
                model.stem.pool = nn.Identity()

    return query_model, model