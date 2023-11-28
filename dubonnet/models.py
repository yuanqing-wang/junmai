import torch

class DubonNet(torch.nn.Module):
    def forward(self, basis, parameters):
        # K.shape = (N, N, N_rbf, N_basis)
        # Q.shape = (N, N, N_rbf, N_basis)
        # W0.shape = (N, N_basis, N_basis)
        # W1.shape = (N, N_basis, 1)
        K, Q, W0, W1 = parameters

        # (N, 3, N_basis)
        K, Q = torch.matmul(basis, K).mean(-3), torch.matmul(basis, Q).mean(-3)

        # (N, N_basis)
        Z = torch.einsum("...ab, ...ab -> ...b", K, Q)

        # (N, N_basis)
        Z = torch.matmul(Z, W0).tanh()
        return Z.sum()
