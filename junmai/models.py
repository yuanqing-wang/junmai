from typing import Optional
import torch
import lightning as pl
from .layers import JunmaiLayer, GaussianDropout
from .semantic import InductiveParameter

class JunmaiModel(pl.LightningModule):
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
    
    def training_step(self, batch, batch_idx):
        E, F, R = batch
        E_hat = self(R)
        loss = torch.nn.functional.mse_loss(E_hat, E)
        print(loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
# Path: junmai/layers.py
    


