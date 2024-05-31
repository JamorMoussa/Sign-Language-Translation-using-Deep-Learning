import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class GCNModelConfigs:
   
    gcn_layers: List[Tuple[int, int]] = field(default_factory=lambda: [(2, 64), (64, 32)])
    fc_layers: List[Tuple[int, int]] = field(default_factory=lambda: [(32, 32)])

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes: int = 28

    @staticmethod
    def get_defaults():
        return GCNModelConfigs()
    
    def to_dict(self):
        fc_layers = [*self.fc_layers, (self.fc_layers[-1][1], self.num_classes)]
        return {"gcn_layers": self.gcn_layers, "fc_layers": fc_layers, "device": str(self.device), "num_classes": self.num_classes}

class GCNModel(nn.Module):

    def __init__(self, configs: GCNModelConfigs = GCNModelConfigs.get_defaults()) -> None:
        super(GCNModel, self).__init__() 
        
        self.configs: GCNModelConfigs = configs

        self.gcn = gnn.Sequential('x, edge_index', [
            (self.gcn_conv_block(in_dim, out_dim), 'x, edge_index -> x') 
            for (in_dim, out_dim) in self.configs.gcn_layers
        ])

        self.fc = nn.Sequential(
            *[self.fc_block(in_dim, out_dim) for (in_dim, out_dim) in self.configs.fc_layers]
        )

        last_dim = self.configs.fc_layers[-1][1]

        self.fc.add_module(name= "out", module=nn.Linear(last_dim , self.configs.num_classes))

    def gcn_conv_block(self, in_dim: int, out_dim: int):
        return gnn.Sequential('x, edge_index', [
            (gnn.GCNConv(in_dim, out_dim), 'x, edge_index -> x'),
            nn.ReLU(True),
        ])

    def fc_block(self, in_dim: int, out_dim: int,):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(True),
        )

    def forward(self, input: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        
        out = self.gcn(input, edge_index)
        out = gnn.global_max_pool(out, batch)
        
        return self.fc(out)
