import torch
import math

NUM_RBF = 50
CUTOFF_LOWER = 1e-12
CUTOFF_UPPER = 5.0
EPSILON = 1e-6
INF = 1e5

def get_x_minus_xt(x):
    return x.unsqueeze(-3) - x.unsqueeze(-2)

def get_x_minus_xt_norm(x_minus_xt):
    x_minus_xt_norm = (
        (x_minus_xt ** 2).sum(-1, keepdims=True) + EPSILON
    ) ** 0.5
    return x_minus_xt_norm

def get_h_cat_ht(h):
    n_nodes = h.shape[-2]
    h_shape = (*h.shape[:-2], n_nodes, n_nodes, h.shape[-1])
    h_cat_ht = torch.cat(
        [
            h.unsqueeze(-3).expand(h_shape),
            h.unsqueeze(-2).expand(h_shape),
        ],
        dim=-1,
    )
    return h_cat_ht

class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff_lower=CUTOFF_LOWER, cutoff_upper=CUTOFF_UPPER):
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
        cutoff_lower=CUTOFF_LOWER,
        cutoff_upper=CUTOFF_UPPER,
        num_rbf=NUM_RBF,
        trainable=False,
        cutoff=True,
    ):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)
        if cutoff:
            self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        else:
            self.cutoff_fn = lambda x: torch.ones_like(x)

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
        ) * self.cutoff_fn(dist)