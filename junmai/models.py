from typing import Optional
import torch
from .layers import JunmaiLayer, GaussianDropout
from .semantic import InductiveParameter

class JunmaiModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_rbf: Optional[int] = None,
        num_particles: Optional[int] = None,
    ):
        super().__init__()
        self.semantic = InductiveParameter(
            num_particles=num_particles,
            in_features=num_rbf,
            out_features=hidden_features,
        )

        self.layer = JunmaiLayer(
            out_features=1,
            hidden_features=hidden_features,
            num_rbf=num_rbf,
        )

    def forward(self, x):
        W = self.semantic()
        return self.layer(x, W)

