import numpy as np
import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger


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

    # from junmai.functional import edge_outer_product, edge_outer_product_jacobian
    # R_EOP = edge_outer_product(R[0])
    # # F_EPO = torch.stack([edge_outer_product_jacobian(_R) for _R in R.unbind(0)], 0)
    # for _R in R.unbind(0):
    #     print(_R.shape)
    #     F_EPO = edge_outer_product_jacobian(_R)
    #     print(F_EPO.shape)

    E_MEAN, E_STD = E.mean(), E.std()
    data_te = np.load(args.test_path)
    E_te, F_te, R_te, Z_te = get_data(data_te)
    E = (E - E_MEAN) / E_STD
    F = F / E_STD
    E_te = (E_te - E_MEAN) / E_STD
    F_te = F_te / E_STD

    R.requires_grad_(True)
    R_te.requires_grad_(True)

    Z = Z.repeat(E.shape[0], 1, 1)

    dataset = torch.utils.data.TensorDataset(E, F, R, Z)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size if args.batch_size > 0 else E.shape[0],
        shuffle=True,
    )
    
    dataset_te = torch.utils.data.TensorDataset(E_te, F_te, R_te, Z)
    dataloader_te = torch.utils.data.DataLoader(
        dataset_te,
        batch_size=args.batch_size if args.batch_size > 0 else E_te.shape[0],
        shuffle=True,
    )

    from junmai.models import JunmaiModel
    model = JunmaiModel(
        in_features=Z.shape[-1],
        hidden_features=args.hidden_features,
        num_rbf=args.num_rbf,
        # num_particles=Z.shape[1],
    )

    trainer = pl.Trainer(
        limit_train_batches=100, 
        max_epochs=10000, 
        log_every_n_steps=1, 
        logger=CSVLogger("logs", name="junmai"),
        devices="auto",
        accelerator="auto",
    )
    trainer.fit(model, dataloader, dataloader_te)

    
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
