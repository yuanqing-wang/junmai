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
class _TuneReportCallback(TuneReportCallback, pl.Callback):
    pass


def get_data(data):
    E, F, R, Z = data["E"], data["F"], data["R"], data["z"]
    E, F, R, Z = torch.from_numpy(E), torch.from_numpy(F), torch.from_numpy(R), torch.from_numpy(Z)
    Z = Z.unsqueeze(0)
    E, F, R, Z = E.to(torch.float32), F.to(torch.float32), R.to(torch.float32), Z.to(torch.float32)
    Z = torch.nn.functional.one_hot(Z.to(torch.int64)).float()
    return E, F, R, Z


def run(args):
    data = np.load(args.path)
    E, F, R, Z = get_data(data)

    E_MEAN, E_STD = E.mean(), E.std()
    data_te = np.load(args.test_path)
    E_te, F_te, R_te, Z_te = get_data(data_te)
    E = (E - E_MEAN) / E_STD
    F = F / E_STD
    E_te = (E_te - E_MEAN) / E_STD
    F_te = F_te / E_STD
    Z = torch.randn_like(Z)
    Z = torch.nn.functional.one_hot(
        torch.arange(Z.shape[-2])
    )
    # Z_te = torch.randn_like(Z_te)
    Z_te = Z

    R.requires_grad_(True)
    R_te.requires_grad_(True)

    dataset = torch.utils.data.TensorDataset(E, F, R)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size if args.batch_size > 0 else E.shape[0],
        shuffle=True,
    )
    
    dataset_te = torch.utils.data.TensorDataset(E_te, F_te, R_te)
    dataloader_te = torch.utils.data.DataLoader(
        dataset_te,
        batch_size=args.batch_size if args.batch_size > 0 else E_te.shape[0],
        shuffle=True,
    )

    def train(config):
        from junmai.models import JunmaiModel
        model = JunmaiModel(
            in_features=Z.shape[-1],
            hidden_features=config["hidden_features"],
            num_rbf=config["num_rbf"],
            num_particles=Z.shape[1],
        )

        trainer = pl.Trainer(
            max_epochs=5000, 
            # log_every_n_steps=1, 
            logger=False,
            devices="auto",
            accelerator="auto",
            callbacks=[_TuneReportCallback()],
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader, dataloader_te)

    configs = {
        "hidden_features": tune.lograndint(16, 32),
        "num_rbf": tune.randint(16, 32),
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-10, 1e-3),
    }

    scheduler = ASHAScheduler(max_t=5000, grace_period=10, reduction_factor=2)

    train = tune.with_resources(train, { "cpu": 1, "gpu": 1 })
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
    parser.add_argument("--path", type=str, default="ethanol_ccsd_t-train.npz")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--test-path", type=str, default="ethanol_ccsd_t-train.npz")
    parser.add_argument("--num-rbf", type=int, default=100)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    run(args)
