import torch

class DubonNet(torch.nn.Module):
    def forward(self, basis, parameters):
        K, Q, W0, W1 = parameters

        # (N, 3, N_basis)
        K, Q = torch.matmul(basis, K).mean(-3), torch.matmul(basis, Q).mean(-3)

        # (N, N_basis)
        Z = torch.tensordot(K, Q, dims=([-2], [-2]))

        # (N, 1)
        Z = (Z @ W0).tanh() @ W1
        return Z.sum()
