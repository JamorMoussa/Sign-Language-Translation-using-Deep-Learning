import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class GCNModelConfigs:
    input_dim: int = 2
    num_classes: int = 28
    gcn_layers: List[Tuple[int, int]] = field(default_factory=lambda: [(2, 64), (64, 32)])
    fc_layers: List[Tuple[int, int]] = field(default_factory=lambda: [(32, 32), (32, 28)])

    @staticmethod
    def get_defaults():
        return GCNModelConfigs()

class GCNModel(nn.Module):

    def __init__(self, configs: GCNModelConfigs = GCNModelConfigs.get_defaults()) -> None:
        super().__init__()  # Ensure proper initialization
        self.configs: GCNModelConfigs = configs

        self.gcn = gnn.Sequential('x, edge_index', [
            (self.gcn_conv_block(in_dim, out_dim), 'x, edge_index -> x') 
            for (in_dim, out_dim) in self.configs.gcn_layers
        ])

        self.fc = nn.Sequential(
            *[self.fc_block(in_dim, out_dim) for (in_dim, out_dim) in self.configs.fc_layers]
        )

    def gcn_conv_block(self, in_dim: int, out_dim: int):
        return gnn.Sequential('x, edge_index', [
            (gnn.GCNConv(in_dim, out_dim), 'x, edge_index -> x'),
            nn.ReLU(True),
        ])

    def fc_block(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(True),
        )

    def forward(self, input: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        
        out = self.gcn(input, edge_index)
        out = gnn.global_max_pool(out, batch)
        
        return self.fc(out)
