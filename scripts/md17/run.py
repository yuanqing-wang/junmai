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
    E_te = (E_te - E_MEAN) / E_STD
    F_te = F_te / E_STD
    Z = torch.randn_like(Z)
    Z_te = torch.randn_like(Z_te)
    
    from junmai.models import JunmaiModel
    model = JunmaiModel(
        in_features=Z.shape[-1],
        hidden_features=args.hidden_features,
        depth=args.depth,
        alpha=args.alpha,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        E = E.cuda()
        F = F.cuda()
        R = R.cuda()
        Z = Z.cuda()
        E_te = E_te.cuda()
        F_te = F_te.cuda()
        R_te = R_te.cuda()
        Z_te = Z_te.cuda()

    # R.requires_grad_(True)
    # R_te.requires_grad_(True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1000,
        verbose=True,
    )

    for i in range(1000000):
        idxs = torch.randperm(E.shape[0])
        batch_size = args.batch_size if args.batch_size > 0 else E.shape[0]
        n_batches = E.shape[0] // batch_size
        for idx_batch in range(n_batches):
            model.train()
            optimizer.zero_grad()
            idx = idxs[idx_batch * batch_size : (idx_batch + 1) * batch_size]
            E_batch = E[idx]
            F_batch = F[idx]
            R_batch = R[idx]
            Z_batch = Z

            E_hat, h_last = model(Z_batch, R_batch)
            # h_last_var = h_last.var(dim=0).mean()

            loss_energy = torch.nn.L1Loss()(E_hat, E_batch)
            F_hat = -1.0 * torch.autograd.grad(
                E_hat.sum(),
                R_batch,
                create_graph=True,
            )[0]

            loss_force = torch.nn.MSELoss()(F_hat, F_batch)     
            loss = 0.001 * loss_energy + loss_force
            loss.backward()
            scheduler.step(loss_energy)
            print(E_STD * loss_energy.item(), loss_force, flush=True)
            optimizer.step()

            # model.eval()
            # E_te_hat, _ = model(Z_te, R_te)
            # F_te_hat = -1.0 * torch.autograd.grad(
            #     E_te_hat.sum(),
            #     R_te,
            #     create_graph=True,
            # )[0]
            # loss_energy_te = torch.nn.L1Loss()(E_te_hat, E_te).item() * E_STD
            # loss_force_te = torch.nn.L1Loss()(F_te_hat, F_te).item() * E_STD
            # print(loss_energy, loss_force, loss_energy_te, loss_force_te, h_last_var, flush=True)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("--path", type=str)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--num-rbf", type=int, default=100)
    parser.add_argument("--hidden-features", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    run(args)
