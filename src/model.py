import torch
import torch.nn as nn
import torchvision.models as models

class QualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_feats = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.regressor = nn.Linear(num_feats, 1)

    def forward(self, x):
        x = self.base(x)
        return self.regressor(x).squeeze(1)
