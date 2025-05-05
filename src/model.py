import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,
                      out_channels = 64,
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64,
                      out_channels = 64,
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64,
                      32,
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,
                      10,
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 10 * 49,
                      out_features = 10)
            )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
