import math
import torch

NUM_RBF = 50
CUTOFF_LOWER = 1e-12
CUTOFF_UPPER = 5.0
EPSILON = 1e-12

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
    
    
class EuclideanAttention(torch.nn.Module):
    def __init__(
            self, 
            cutoff_lower=CUTOFF_LOWER,
            cutoff_upper=CUTOFF_UPPER,
            num_rbf=NUM_RBF,
        ):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.register_buffer(
            "gamma", 
            torch.linspace(cutoff_lower, cutoff_upper, num_rbf),
        )
        


    def forward(self, dist):
        return torch.softmax(-dist / self.gamma, -2)

class BasisGeneration(torch.nn.Module):
    def __init__(
        self,
        attention,
    ):
        super().__init__()
        self.attention = attention

    def forward(self, x):
        # (N, N, 3)
        delta_x = x[..., None, :, :] - x[..., :, None, :]

        # (N, N, 1)
        delta_x_norm = delta_x.pow(2).relu().sum(dim=-1, keepdim=True)

        # (N, N, N_rbf)
        delta_x_attention = self.attention(delta_x_norm)

        # (N, N, 3, N_rbf)
        basis = delta_x.unsqueeze(-1) * delta_x_attention.unsqueeze(-2)

        return basis

class ParameterGeneration(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        num_basis,
        num_rbf=NUM_RBF,
        num_heads=1,
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.num_basis = num_basis
        self.fc = torch.nn.Linear(in_features, hidden_features)
        self.attn = torch.nn.MultiheadAttention(hidden_features, num_heads)
        self.interaction_out_features = (
            num_rbf * num_basis,
            num_rbf * num_basis,
        )
        self.node_out_features = (
            num_basis * num_basis,
            num_basis,
            num_basis,
        )
        self.fc_interaction = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_features, hidden_features),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_features, sum(self.interaction_out_features)),
        )
        
        self.fc_node = torch.nn.Linear(hidden_features, sum(self.node_out_features))

    def forward(self, h):
        h = self.fc(h)
        h, _ = self.attn(h, h, h)
        n_nodes = int(h.shape[-2])
        h_interaction = torch.cat(
            [
                h.unsqueeze(-3).repeat_interleave(n_nodes, -3),
                h.unsqueeze(-2).repeat_interleave(n_nodes, -2),
            ],
            dim=-1
        )
        K, Q = torch.split(self.fc_interaction(h_interaction), self.interaction_out_features, dim=-1)

        # (N, N, N_rbf, N_basis)
        W0, B0, W1 = torch.split(self.fc_node(h), self.node_out_features, dim=-1)

        # (N, N, N_rbf, N_basis)
        K = K.reshape(*K.shape[:-1], self.num_rbf, self.num_basis)

        # (N, N, N_rbf, N_basis)
        Q = Q.reshape(*Q.shape[:-1], self.num_rbf, self.num_basis)

        # (N, N_basis)
        W0 = W0.reshape(*W0.shape[:-1], self.num_basis, self.num_basis)
        B0 = B0.reshape(*B0.shape[:-1], self.num_basis)
        W1 = W1.reshape(*W1.shape[:-1], self.num_basis, 1)
        return K, Q, W0, B0, W1
    





