import torch.nn as nn
import torch.nn.functional as F
from models import create_model

class MetricModel(nn.Module):
    def __init__(self, modelname: str, pretrained: bool = False, simclr_dim: int = 128, **model_params):
        super().__init__()
        
        _, self.model = create_model(modelname=modelname, pretrained=pretrained, **model_params)
        self.num_features = self.model.num_features

        self.simclr_layer = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, simclr_dim),
        )
        self.shift_cls_layer = nn.Linear(self.num_features, 4)

    
    def forward_features(self, x):
        features = self.model(x)
        features = F.normalize(self.simclr_layer(features), dim=1)
        
        return features
    
    def forward(self, x, shift=False):        
        
        outputs = {}
        
        features = self.model(x)
        outputs['simclr'] = self.simclr_layer(features)

        if shift:
            outputs['shift'] = self.shift_cls_layer(features)
        
        return outputs