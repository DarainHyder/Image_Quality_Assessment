import torch
import torch.nn as nn
import torchvision.models as models

class QualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # EfficientNet stores its final linear layer in the 'classifier' sequential block
        num_feats = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.base = base
        # Replacing the classification head with a regression head
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_feats, 1)
        )

    def forward(self, x):
        x = self.base(x)
        return self.regressor(x).squeeze(1)
