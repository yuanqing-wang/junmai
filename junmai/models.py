import torch
from .layers import JunmaiLayer

class JunmaiModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        depth: int,
        activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                JunmaiLayer(
                    in_features if i == 0 else hidden_features,
                    hidden_features if i < depth - 1 else 1,
                )
                for i in range(depth)
            ]
        )

    def forward(self, h, x):
        for layer in self.layers:
            h = layer(h, x)
        return h
