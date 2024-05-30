from dataclasses import dataclass, field
import torch
import torch.nn as nn
from typing import Tuple, List

__all__ = ["MLPConfigs", "MLPModel"]

@dataclass
class MLPConfigs:
    input_size: int = 42
    hidden_size: int = 64
    num_classes: int = 28
    mlp_layers: List[Tuple[int, int]] = field(default_factory=lambda: [(42, 64), (64, 28)])

    @staticmethod
    def get_defaults():
        return MLPConfigs()

class MLPModel(nn.Module):
    def __init__(self, configs: MLPConfigs = MLPConfigs.get_defaults()) -> None:
        super(MLPModel, self).__init__()
        self.configs = configs
        

        self.mlp = nn.Sequential(*[self.mlp_block(input_size, output_size) for input_size, output_size in self.configs.mlp_layers[:-1]])

       
        _, last_hidden_size = self.configs.mlp_layers[-1]
        self.output_layer = nn.Linear(last_hidden_size, self.configs.num_classes)

    def mlp_block(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.output_layer(x)
        return x

# Example usage
mlp_configs = MLPConfigs()
print(mlp_configs.hidden_size)  # This will print the default hidden_size which is 64

mlp_model = MLPModel(mlp_configs)
print(mlp_model)
