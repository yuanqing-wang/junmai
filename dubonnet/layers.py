import math
from re import L
import torch


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (
                torch.cos(distances * math.pi / self.cutoff_upper) + 1.0
            )
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class ExpNormalSmearing(torch.nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        cutoff=False,
    ):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)
        if cutoff:
            self.cutoff_fn = CosineCutoff(0, cutoff_upper)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", torch.nn.Parameter(means))
            self.register_parameter("betas", torch.nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

        self.out_features = self.num_rbf

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        return torch.exp(
            -self.betas
            * (
                torch.exp(self.alpha * (-dist + self.cutoff_lower))
                - self.means
            )
            ** 2
        )

class BasisGeneration(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        num_rbf,
        num_basis,
    ):
        super(BasisGeneration, self).__init__()
        self.fc = torch.nn.Linear(2 * in_features, hidden_features)


    def forward(self, h):
        pass





class DubonNet(torch.nn.Module):
    def __init__(
        self,
        in_features,
        num_rbf,
        num_basis,
    ):
        super(DubonNet, self).__init__()
        self.in_features = in_features
        self.smearing = ExpNormalSmearing(num_rbf=num_rbf)
        self.basis_generation = BasisGeneration(
            in_features=in_features, num_rbf=num_rbf, num_basis=num_basis,
        )

    def forward(self, h, x):
        # (N, N, 3)
        delta_x = x[..., None, :, :] - x[..., :, None, :]

        # (N, N, 1)
        delta_x_norm = torch.norm(delta_x, dim=-1).unsqueeze(-1)

        # (N, N, N_rbf)
        delta_x_smeared = self.smearing(delta_x_norm)

        # (N, N, 3)
        delta_x_unit = delta_x / delta_x_norm

        # (N, N, 3, N_rbf)
        basis = delta_x_unit.unsqueeze(-1) * delta_x_smeared.unsqueeze(-2)

        # (N, N, N_rbf, N_basis)
        K, Q, W0, W1 = self.basis_generation(h)

        # (N, N, 3, N_basis)
        K, Q = torch.matmul(basis, K), torch.matmul(basis, Q)

        # (N, N, N_basis)
        Z = torch.tensordot(K, Q, dims=([-2], [-2]))

        # (N, N, 1)
        Z = (Z @ W0).tanh() @ W1
        return Z.sum()





