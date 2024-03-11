import numpy as np
import torch
from junmai.layers import ExpNormalSmearing

def get_data(data):
    E, F, R, Z = data["E"], data["F"], data["R"], data["z"]
    E, F, R, Z = torch.from_numpy(E), torch.from_numpy(F), torch.from_numpy(R), torch.from_numpy(Z)
    Z = Z.unsqueeze(0)
    E, F, R, Z = E.to(torch.float32), F.to(torch.float32), R.to(torch.float32), Z.to(torch.float32)
    Z = torch.nn.functional.one_hot(Z.to(torch.int64)).float()
    return E, F, R, Z

class Model(torch.nn.Module):
    def __init__(self, in_features, hidden_features, n_basis=16):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_basis * (in_features ** 2), hidden_features),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_features, 1),
        )

        self.K = torch.nn.Parameter(torch.randn(in_features, in_features, n_basis))
        self.Q = self.K
        # self.Q = torch.nn.Parameter(1e-3 * torch.randn(in_features, in_features, n_basis))
        # self.smearing = ExpNormalSmearing()
    
    def forward(self, x):
        delta_x = x.unsqueeze(-2) - x.unsqueeze(-3)
        delta_x = delta_x.unsqueeze(-2)
        k = self.K.unsqueeze(-1) * delta_x
        q = self.Q.unsqueeze(-1) * delta_x
        xxt = torch.einsum("...ij,...ij->...i", k, q)
        xxt = xxt.flatten(-3, -1)
        return self.fc(xxt)

def run():
    data = np.load("ethanol_ccsd_t-train.npz")
    E, F, R, Z = get_data(data)
    E, F, R = E[:100], F[:100], R[:100] 
    R.requires_grad_(True)
    E_MEAN, E_STD = E.mean(), E.std()
    E = (E - E_MEAN) / E_STD
    F = F / E_STD

    model = Model(
        in_features=Z.shape[-2],
        hidden_features=16,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=100,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        E = E.cuda()
        F = F.cuda()
        R = R.cuda()
        Z = Z.cuda()

    for i in range(100000):
        optimizer.zero_grad()
        E_pred = model(R)
        # loss = torch.nn.functional.mse_loss(E_pred, E)
        # loss.backward()
        F_pred = torch.autograd.grad(-E_pred.sum(), R, create_graph=True)[0]
        loss = torch.nn.functional.mse_loss(F_pred, F)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print(loss.item())




if __name__ == "__main__":
    run()
    
