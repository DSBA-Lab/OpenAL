from timm import create_model as _create_model


def create_model(modelname: str, num_classes: int, img_size: int, pretrained: bool = False, **kwargs):
    if modelname in __import__('models').__dict__.keys():
        return __import__('models').__dict__[modelname](num_classes=num_classes, img_size=img_size, **kwargs)
    else:
        return _create_model(modelname, num_classes=num_classes, pretrained=pretrained, **kwargs)

        