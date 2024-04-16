import numpy as np
import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger

def run(args):
    from junmai.data.ccsd import CCSD
    data = CCSD(args.name, batch_size=args.batch_size, normalize=True)
    data.setup()
    print(data.E_STD)

    from junmai.models import JunmaiModel
    model = JunmaiModel(
        in_features=9,
        hidden_features=args.hidden_features,
        num_rbf=args.num_rbf,
        # num_particles=9,
        lr=args.lr,
        weight_decay=args.weight_decay,
        E_MEAN=data.E_MEAN,
        E_STD=data.E_STD,
    )

    # from lightning.pytorch.callbacks import LearningRateMonitor
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        # limit_train_batches=100, 
        max_epochs=1000000, 
        log_every_n_steps=1, 
        logger=CSVLogger("logs", name="junmai"),
        devices="auto",
        accelerator="auto",
        enable_progress_bar=False,
    )
    trainer.fit(model, data)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("--name", type=str, default="ethanol")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num-rbf", type=int, default=50)
    parser.add_argument("--hidden-features", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run(args)
