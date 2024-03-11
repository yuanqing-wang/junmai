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
    def __init__(self, in_features, hidden_features):
        super().__init__()

        self.fc = torch.nn.Sequential(
            # torch.nn.Linear(in_features, hidden_features),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_features, 1),
        )

        self.smearing = ExpNormalSmearing()
        self.after_smearing = torch.nn.Linear(self.smearing.num_rbf, hidden_features)
        self.W = torch.nn.Parameter(
            torch.randn(
                9, 9, 9, 9, hidden_features, hidden_features,
            ),
        )

    def forward(self, x):
        delta_x = x.unsqueeze(-2) - x.unsqueeze(-3)
        delta_x_norm = delta_x.pow(2).sum(-1, keepdims=True)
        delta_x_norm_smeared = self.smearing(delta_x_norm)
        delta_x_norm_smeared = self.after_smearing(delta_x_norm_smeared)
        delta_x = delta_x / (delta_x_norm + 1e-6)
        delta_x = delta_x.unsqueeze(-2) * delta_x_norm_smeared.unsqueeze(-1)
        print(delta_x.shape)
        delta_x = torch.einsum("...abkc, ...dekc, abdeko -> ...o", delta_x, delta_x, self.W)
        return self.fc(delta_x)

def run():
    data = np.load("ethanol_ccsd_t-train.npz")
    E, F, R, Z = get_data(data)
    E, F, R = E[:100], F[:100], R[:100]
    R.requires_grad = True
    E_MEAN, E_STD = E.mean(), E.std()
    E = (E - E_MEAN) / E_STD
    F = F / E_STD

    model = Model(
        in_features=(Z.shape[-2] ** 4) * 32,
        hidden_features=32,
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
        scheduler.step(loss)
        optimizer.step()
        print(loss.item())




if __name__ == "__main__":
    run()
    
