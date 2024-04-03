from torchlars import LARS

def create_optimizer(opt_name: str, model, lr: float, opt_params: dict = {}, backbone: bool = False):
    # optimizer
    if backbone:
        optimizer = {}
        if getattr(model, 'LPM', False):
            optimizer['LPM'] = _create_optimizer(opt_name=opt_name, model=model.LPM, lr=lr, opt_params=opt_params)
            optimizer['backbone'] = _create_optimizer(opt_name=opt_name, model=model.backbone, lr=lr, opt_params=opt_params)
        else:
            optimizer['backbone'] = _create_optimizer(opt_name=opt_name, model=model, lr=lr, opt_params=opt_params)
    else:
        optimizer = _create_optimizer(opt_name=opt_name, model=model, lr=lr, opt_params=opt_params)
            
    return optimizer


def _create_optimizer(opt_name: str, model, lr: float, opt_params: dict = {}):
    lars_params = opt_params.pop('LARS', None)
    
    # optimizer
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name](params=model.parameters(), lr=lr, **opt_params)

    if lars_params != None:
        # apply LARS
        optimizer = LARS(optimizer, **lars_params)
            
    return optimizer