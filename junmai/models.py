from typing import Optional
import torch
import lightning as pl
import ray
from .layers import JunmaiLayer, GaussianDropout
from .semantic import InductiveParameter

class JunmaiModel(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_rbf: Optional[int] = None,
        num_particles: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
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
        self.validation_step_outputs = []
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        W = self.semantic()
        return self.layer(x, W)
    
    def training_step(self, batch, batch_idx):
        E, F, R = batch
        E_hat = self(R)
        F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        loss_energy = torch.nn.functional.mse_loss(E_hat, E)
        loss_force = torch.nn.functional.mse_loss(F_hat, F)
        loss = 1e-3 * loss_energy + loss_force
        return loss
    
    def validation_step(self, batch, batch_idx):
        E, F, R = batch
        R.requires_grad_(True)
        # enable grad
        with torch.set_grad_enabled(True):
            E_hat = self(R)
            F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        loss_energy = torch.nn.functional.l1_loss(E_hat, E)
        loss_force = torch.nn.functional.l1_loss(F_hat, F)
        self.validation_step_outputs.append((loss_energy, loss_force))

    def on_validation_epoch_end(self):
        loss_energy, loss_force = zip(*self.validation_step_outputs)
        loss_energy = torch.stack(loss_energy).mean()
        loss_force = torch.stack(loss_force).mean()
        self.log("val_loss_energy", loss_energy)
        self.log("val_loss_force", loss_force)
        # ray.train.report({"loss_energy": loss_energy.item(), "loss_force": loss_force.item()})
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    
    


