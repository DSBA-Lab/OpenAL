try:
    from torchlars import LARS
    is_torchlars = True
except:
    is_torchlars = False
    print('To use torchlars, torchlars should be installed.')

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

    if (lars_params != None) and is_torchlars:
        # apply LARS
        optimizer = LARS(optimizer, **lars_params)
    elif is_torchlars == False:
        print('LARS cennot be used without torchlars installation.')
            
    return optimizer