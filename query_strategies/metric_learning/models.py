import torch.nn as nn
import torch.nn.functional as F

from timm.layers import ClassifierHead

class MetricModel(nn.Module):
    def __init__(self, backbone, simclr_dim=128):
        super().__init__()
        
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.head = ClassifierHead(num_features=self.num_features, num_classes=0)
        
        self.simclr_layer = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, simclr_dim),
        )
        self.shift_cls_layer = nn.Linear(self.num_features, 4)

    
    def forward_features(self, x):
        features = self.head(self.backbone.forward_features(x))
        features = F.normalize(self.simclr_layer(features), dim=1)
        
        return features
    
    def forward(self, x, shift=False):        
        
        features = self.forward_features(x)

        if shift:
            outputs = {'features': features}
            outputs['shift'] = self.shift_cls_layer(features)
        else:
            outputs = features
        
        return outputs