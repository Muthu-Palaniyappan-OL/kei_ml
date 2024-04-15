import torch
import torch.nn as nn
from torchsummary import summary


class Model(nn.Module):

    def __init__(self, output_dim=11):
        super().__init__()
        self.shufflenet = torch.hub.load(
            "pytorch/vision:v0.10.0", "shufflenet_v2_x1_0", pretrained=True
        )
        self.output_dim = output_dim
        self.model = nn.Sequential(
            self.shufflenet,
            nn.Linear(1000, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, self.output_dim * self.output_dim),
        )

    def forward(self, x):
        return self.model(x)
