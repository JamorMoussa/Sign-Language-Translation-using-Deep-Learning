## TODO: create MLPConfigs + MLPModel 

from dataclasses import dataclass
import torch, torch.nn as nn

__all__ = ["MLPConfigs", "MLPModel"]

@dataclass
class MLPConfigs:

    input_size: int = 3
    size: int = 10
    ld: float = 1.0

    @staticmethod
    def get_defaults(self):
        return MLPConfigs()


class MLPModel(nn.Module):

    configs: MLPConfigs

    def __init__(self, configs: MLPConfigs = MLPConfigs.get_defaults()):
        self.configs = configs

        nn.Sequential(
            nn.Linear(self.configs.input_size, )
        )


