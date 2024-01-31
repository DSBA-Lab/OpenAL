from torchlars import LARS

def create_optimizer(opt_name: str, model, lr: float, opt_params: dict = {}):
    lars_params = opt_params.pop('LARS', None)
    
    # optimizer
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name](params=model.parameters(), lr=lr, **opt_params)

    if lars_params != None:
        # apply LARS
        optimizer = LARS(optimizer, **lars_params)
            
    return optimizer