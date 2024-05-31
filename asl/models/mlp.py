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
    
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_defaults():
        return MLPConfigs()
    def to_dict(self):
        return {"MLP_layers": self.mlp_layers,  "device": str(self.device), "num_classes": self.num_classes}

    

class MLPModel(nn.Module):
    def __init__(self, configs: MLPConfigs = MLPConfigs.get_defaults()) -> None:
        super(MLPModel, self).__init__()
        self.configs = configs
        

        self.mlp = nn.Sequential(*[self.mlp_block(input_size, output_size) for input_size, output_size in self.configs.mlp_layers[:-1]])

       
        last, num_classes = self.configs.mlp_layers[-1]
        self.output_layer = nn.Linear(last , num_classes)

    def mlp_block(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.output_layer(x)
        return x

mlp_configs = MLPConfigs.get_defaults()
print(mlp_configs)

mlp_configs.mlp_layers = [(30, 128), (128, 28)]
ml  = MLPModel(configs= mlp_configs)
print(ml)

