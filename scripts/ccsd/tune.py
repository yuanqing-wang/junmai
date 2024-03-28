import os
import numpy as np
import torch
import lightning as pl
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.ax import AxSearch
from ray.tune.search.optuna import OptunaSearch
from lightning.pytorch.loggers import CSVLogger
class _TuneReportCallback(TuneReportCallback, pl.Callback):
    pass



def run(args):
    from junmai.data.dataset import MD17

    def train(args):
        args = argparse.Namespace(**args)
        data = MD17(args.name, batch_size=args.batch_size)
        data.setup()
        from junmai.models import JunmaiModel
        model = JunmaiModel(
            in_features=9,
            hidden_features=args.hidden_features,
            num_rbf=args.num_rbf,
            num_particles=9,
            E_MEAN=data.E_MEAN,
            E_STD=data.E_STD,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        trainer = pl.Trainer(
            limit_train_batches=100, 
            max_epochs=10000, 
            log_every_n_steps=1, 
            logger=CSVLogger("logs", name="junmai"),
            devices="auto",
            accelerator="auto",
            enable_progress_bar=False,
        )
        trainer.fit(model, data)

    configs = {
        "name": "ethanol",
        "hidden_features": tune.lograndint(16, 32),
        "num_rbf": tune.randint(16, 32),
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-10, 1e-3),
        "batch_size": tune.lograndint(8, 64),
    }

    scheduler = ASHAScheduler(max_t=5000, grace_period=10, reduction_factor=2)

    train = tune.with_resources(train, { "cpu": 2, "gpu": 1 })
    tuner = tune.Tuner(
        train,
        param_space=configs,
        tune_config=tune.TuneConfig(
            metric="val_loss_energy",
            mode="min",
            num_samples=1000,
            scheduler=scheduler,
            search_alg=OptunaSearch(),
        ),
        run_config=RunConfig(
            storage_path=os.path.join(os.getcwd(), "ray_results"),
        ),
    )
    return tuner.fit()


    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MD simulation")
    args = parser.parse_args()
    run(args)
