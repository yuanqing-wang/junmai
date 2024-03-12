import torch

class InductiveParameter(torch.nn.Module):
    def __init__(
            self,
            num_particles: int,
            in_features: int,
            out_features: int,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.randn(
                num_particles, 
                num_particles, 
                num_particles,
                num_particles,
                in_features, 
                out_features,
            ),
        )

    def forward(self, ):
        return self.W