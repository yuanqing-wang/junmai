from typing import Optional
import torch
import lightning as pl
import ray
from .layers import JunmaiLayer, GaussianDropout, InductiveParameter, TransductiveParameter

class JunmaiModel(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_rbf: Optional[int] = None,
        num_particles: Optional[int] = None,
        alpha: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        factor: float = 0.5,
        patience: int = 10,
        E_MEAN: float = 0.0,
        E_STD: float = 1.0,
    ):
        super().__init__()

        if num_particles is not None:
            self.semantic = InductiveParameter(
                num_particles=num_particles,
                num_rbf=num_rbf,
                out_features=hidden_features,
            )
        else:
            self.semantic = TransductiveParameter(
                in_features=in_features,
                out_features=hidden_features,
                hidden_features=hidden_features,
                num_rbf=num_rbf,
            )
        
        self.dropout = GaussianDropout(alpha=alpha)
        self.layer = JunmaiLayer(
            out_features=1,
            hidden_features=hidden_features,
            num_rbf=num_rbf,
        )
        self.validation_step_outputs = []
        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.E_MEAN = E_MEAN
        self.E_STD = E_STD

    def forward(self, x, h):
        K, Q = self.semantic(x.detach(), h)
        return self.layer(x, (K, Q)).sum(-2)
    
    def training_step(self, batch, batch_idx):
        R, E, F, Z = batch
        R.requires_grad_(True)
        E_hat = self(R, Z)
        F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        loss_energy = torch.nn.functional.mse_loss(E_hat, E)
        self.log("train_loss_energy", loss_energy)
        loss_force = torch.nn.functional.mse_loss(F_hat, F)
        self.log("train_loss_force", loss_force)
        loss = 1e-2 * loss_energy + loss_force
        return loss
    
    def validation_step(self, batch, batch_idx):
        R, E, F, Z = batch
        R.requires_grad_(True)
        with torch.set_grad_enabled(True):
            E_hat = self(R, Z)
            F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        E_hat = E_hat * self.E_STD + self.E_MEAN
        F_hat = F_hat * self.E_STD
        loss_energy = torch.nn.functional.l1_loss(E_hat, E)
        loss_force = torch.nn.functional.l1_loss(F_hat, F)
        self.validation_step_outputs.append((loss_energy, loss_force))

    def on_validation_epoch_end(self):
        loss_energy, loss_force = zip(*self.validation_step_outputs)
        loss_energy = torch.stack(loss_energy).mean()
        loss_force = torch.stack(loss_force).mean()
        self.log("val_loss_energy", loss_energy)
        self.log("val_loss_force", loss_force)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode="min", 
        #     factor=self.factor, 
        #     patience=self.patience, 
        #     min_lr=1e-6,
        #     verbose=True,
        # )

        # scheduler = {
        #     "scheduler": scheduler,
        #     "monitor": "val_loss_energy",
        # }
    
        # return [optimizer], [scheduler]
        return optimizer
    
    


