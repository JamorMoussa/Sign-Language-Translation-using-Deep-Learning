import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class CNNTinyVGGModelConfigs:
    input_channels: int = 3
    filters: int = 10
    conv_layers: List[Tuple[int, int, int, int]] = field(default_factory=lambda: [
        (3, 10, 3, 1), 
        (10, 10, 3, 1),
        (10, 10, 3, 1),
        (10, 10, 3, 1)
    ])
    fc_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    num_classes: int = 26
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_defaults():
        return CNNTinyVGGModelConfigs()

    def to_dict(self):
        fc_layers = self.fc_layers + [self.num_classes]
        return {
            "input_channels": self.input_channels,
            "filters": self.filters,
            "conv_layers": self.conv_layers,
            "fc_layers": fc_layers,
            "num_classes": self.num_classes,
            "device": str(self.device)
        }

class CNNTinyVGG(nn.Module):
    def __init__(self, configs: CNNTinyVGGModelConfigs = CNNTinyVGGModelConfigs.get_defaults()):
        super(CNNTinyVGG, self).__init__()
        self.configs = configs

        self.cnn_1 = nn.Sequential(
            self.cnn_block_1(self.configs.conv_layers[0][0], self.configs.conv_layers[0][1], self.configs.conv_layers[0][2], self.configs.conv_layers[0][3]),
            self.cnn_block_2(self.configs.conv_layers[1][0], self.configs.conv_layers[1][1], self.configs.conv_layers[1][2], self.configs.conv_layers[1][3]),
            self.cnn_block_1(self.configs.conv_layers[2][0], self.configs.conv_layers[2][1], self.configs.conv_layers[2][2], self.configs.conv_layers[2][3]),
            self.cnn_block_2(self.configs.conv_layers[3][0], self.configs.conv_layers[3][1], self.configs.conv_layers[3][2], self.configs.conv_layers[3][3])
        )

        self.flatten = nn.Flatten()

        fc_input_dim = self.configs.filters * 32 * 32
        self.fc = nn.Sequential(
            self.fc_block(fc_input_dim, self.configs.fc_layers[0]),
            self.fc_block(self.configs.fc_layers[0], self.configs.fc_layers[1]),
            self.fc_block(self.configs.fc_layers[1], self.configs.fc_layers[2]),
            nn.Linear(self.configs.fc_layers[2], self.configs.num_classes)
        )

    def cnn_block_1(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True)
        )
    
    def cnn_block_2(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        return nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def fc_block(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_1(x)
        x = self.flatten(x)
        return self.fc(x)

