import torch

class Junmai(torch.nn.Module):
    def forward(self, basis, parameters):
        print(basis)
        # basis.shape = (N, N, 3, N_rbf)
        # K.shape = (N, N, N_rbf, N_basis)
        # Q.shape = (N, N, N_rbf, N_basis)
        # W0.shape = (N, N_basis, N_basis)
        # W1.shape = (N, N_basis, 1)
        K, Q, W0, B0, W1 = parameters

        # (N, N, 3, N_basis)
        K, Q = torch.matmul(basis, K), torch.matmul(basis, Q)

        # (N, 3, N_basis)
        K, Q = K.mean(-3), Q.mean(-3)

        # (N, N_basis)
        Z = torch.einsum("...ab, ...ab -> ...b", K, Q)

        # (N, N_basis)
        Z = torch.einsum("...ba, ...b -> ...a", W0, Z)
        Z = Z + B0
        Z = Z.sigmoid()
        Z = torch.einsum("...ba, ...b -> ...a", W1, Z)
        return Z.sum(-2)
