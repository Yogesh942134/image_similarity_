import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.head = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.head(x)
        return F.normalize(x, dim=1)
