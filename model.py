import torch.nn as nn
from torchvision.models import resnet18


class MatrixSSLResNet(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        # 1. Load ResNet18 base
        base = resnet18(weights=None)

        # 2. Modify for small CIFAR images (32x32)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()  # Maxpool is too aggressive for 32x32

        # 3. Everything up to the global average pool
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # 4. The Projector (The MLP head where the Matrix loss happens)
        # This is crucial for SSL performance
        self.projector = nn.Sequential(
            nn.Linear(512, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, feature_dim)
        )

    def forward(self, x):
        h = self.backbone(x).flatten(1)
        z = self.projector(h)
        return z
