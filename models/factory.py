from timm import create_model as _create_model

def create_model(modelname: str, num_classes: int, img_size: int, pretrained: bool = False):
    if pretrained:
        return _create_model(modelname, num_classes=num_classes, pretrained=True)
    else:
        return __import__('models').__dict__[modelname](num_classes=num_classes, img_size=img_size)