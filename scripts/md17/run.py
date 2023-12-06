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
    
    from sakelite.layers import (
        BasisGeneration, ExpNormalSmearing, ParameterGeneration
    )

    smearing = ExpNormalSmearing(num_rbf=args.num_rbf)
    basis_generation = BasisGeneration(smearing)
    parameter_generation = ParameterGeneration(
        in_features=Z.shape[-1],
        hidden_features=args.hidden_features,
        num_rbf=args.num_rbf,
        num_basis=args.hidden_features,
        num_heads=1,
    )

    from sakelite.models import SakeLite
    sakelite = SakeLite()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    E, F, R, Z = E.to(device), F.to(device), R.to(device), Z.to(device)
    E_te, F_te, R_te, Z_te = E_te.to(device), F_te.to(device), R_te.to(device), Z_te.to(device)
    basis_generation = basis_generation.to(device)
    parameter_generation = parameter_generation.to(device)
    sakelite = sakelite.to(device)

    optimizer = torch.optim.Adam(
        list(basis_generation.parameters())
        + list(parameter_generation.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for i in range(1000000):
        optimizer.zero_grad()
        basis = basis_generation(R)
        K, Q, W0, B0, W1 = parameter_generation(Z)
        E_hat = sakelite(basis, (K, Q, W0, B0, W1))
        # loss_energy = torch.nn.L1Loss()(E_hat, E)
        F_hat = -1.0 * torch.autograd.grad(
            E_hat.sum(),
            R,
            create_graph=True,
        )[0]

        loss_force = torch.nn.L1Loss()(F_hat, F)
        print(loss_force.item())
        # loss = 0.001 * loss_energy + loss_force
        loss = loss_force

        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     E_hat_te = sakelite(basis_generation(R_te), (K, Q, W0, W1))
        #     E_hat_te = E_hat_te * E_STD + E_MEAN
        #     loss_te = torch.nn.L1Loss()(E_hat_te, E_te)
        #     print(loss_energy.item() * E_STD.item(), loss_te.item())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("--path", type=str)
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--num-rbf", type=int, default=50)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    args = parser.parse_args()
    run(args)
