from fcntl import F_DUPFD
import numpy as np
import torch

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
    R.requires_grad_(True)
    E_MEAN, E_STD = E.mean(), E.std()
    data_te = np.load(args.test_path)
    E_te, F_te, R_te, Z_te = get_data(data_te)
    E = (E - E_MEAN) / E_STD
    F = F / E_STD
    
    from junmai.models import JunmaiModel
    model = JunmaiModel(
        in_features=Z.shape[-1],
        hidden_features=args.hidden_features,
        depth=args.depth,
    )

    for i in range(1000000):
        idxs = torch.randperm(E.shape[0])
        batch_size = args.batch_size if args.batch_size > 0 else E.shape[0]
        n_batches = E.shape[0] // batch_size
        for idx_batch in range(n_batches):
            optimizer.zero_grad()
            idx = idxs[idx_batch * batch_size : (idx_batch + 1) * batch_size]
            E_batch = E[idx]
            F_batch = F[idx]
            R_batch = R[idx]
            Z_batch = Z
            E_hat = model(Z_batch, R_batch)


            loss_energy = torch.nn.L1Loss()(E_hat, E_batch)
            # F_hat = -1.0 * torch.autograd.grad(
            #     E_hat.sum(),
            #     R_batch,
            #     create_graph=True,
            # )[0]

            # loss_force = torch.nn.L1Loss()(F_hat, F_batch)    
            loss = loss_energy
            loss.backward()
            optimizer.step()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("--path", type=str)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--num-rbf", type=int, default=100)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args)
