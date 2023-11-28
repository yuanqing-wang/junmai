import numpy as np
import torch

def run(args):
    data = np.load(args.path)
    # print(list(data.keys()))
    # energy, force, coordinates, species
    E, F, R, Z = data["E"], data["F"], data["R"], data["z"]
    E, F, R, Z = torch.from_numpy(E), torch.from_numpy(F), torch.from_numpy(R), torch.from_numpy(Z)
    Z = Z.unsqueeze(0)
    E, F, R, Z = E.to(torch.float32), F.to(torch.float32), R.to(torch.float32), Z.to(torch.float32)
    Z = torch.nn.functional.one_hot(Z.to(torch.int64)).float()
    E_MEAN, E_STD = E.mean(), E.std()
    E = (E - E_MEAN) / E_STD

    from dubonnet.layers import (
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

    from dubonnet.models import DubonNet
    dubonnet = DubonNet()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    E, F, R, Z = E.to(device), F.to(device), R.to(device), Z.to(device)
    basis_generation = basis_generation.to(device)
    parameter_generation = parameter_generation.to(device)
    dubonnet = dubonnet.to(device)

    optimizer = torch.optim.Adam(
        list(basis_generation.parameters())
        + list(parameter_generation.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for i in range(100000):
        optimizer.zero_grad()
        basis = basis_generation(R)
        K, Q, W0, W1 = parameter_generation(Z)
        E_hat = dubonnet(basis, (K, Q, W0, W1))
        loss = torch.nn.functional.mse_loss(E_hat, E)
        loss.backward()
        optimizer.step()
        print(loss.item() * E_STD.item())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("--path", type=str)
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--num-rbf", type=int, default=50)
    parser.add_argument("--hidden-features", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    args = parser.parse_args()
    run(args)
