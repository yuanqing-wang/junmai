import numpy as np
import torch

def get_data(data):
    E, F, R, Z = data["E"], data["F"], data["R"], data["z"]
    E, F, R, Z = torch.from_numpy(E), torch.from_numpy(F), torch.from_numpy(R), torch.from_numpy(Z)
    Z = Z.unsqueeze(0)
    E, F, R, Z = E.to(torch.float32), F.to(torch.float32), R.to(torch.float32), Z.to(torch.float32)
    Z = torch.nn.functional.one_hot(Z.to(torch.int64)).float()
    return E, F, R, Z

class Model(torch.nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_features, 1),
        )
    
    def forward(self, x):
        xxt = torch.einsum("nij,nkj->nik", x, x)
        xxt = xxt.flatten(-2, -1)
        return self.fc(xxt)

def run():
    data = np.load("ethanol_ccsd_t-train.npz")
    E, F, R, Z = get_data(data)
    E, F, R = E[:100], F[:100], R[:100] 
    R.requires_grad_(True)

    R.requires_grad_(True)
    E_MEAN, E_STD = E.mean(), E.std()
    E = (E - E_MEAN) / E_STD
    F = F / E_STD

    model = Model(
        in_features=Z.shape[-2] ** 2,
        hidden_features=100,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    for i in range(100000):
        optimizer.zero_grad()
        E_pred = model(R)
        # loss = torch.nn.functional.mse_loss(E_pred, E)
        # loss.backward()
        F_pred = torch.autograd.grad(-E_pred.sum(), R, create_graph=True)[0]
        loss = torch.nn.functional.mse_loss(F_pred, F)
        loss.backward()
        optimizer.step()
        print(loss.item())




if __name__ == "__main__":
    run()
    
