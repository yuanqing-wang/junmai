from typing import Optional
import math
import torch
from .utils import (
    ExpNormalSmearing,
    get_h_cat_ht,
    get_x_minus_xt,
    get_x_minus_xt_norm,
    EPSILON,
)

class JunmaiLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_coefficients: Optional[int] = None,
        smearing: torch.nn.Module = ExpNormalSmearing,
    ):
        super().__init__()
        if num_coefficients is None:
            num_coefficients = in_features
        self.num_coefficients = num_coefficients
        self.smearing = smearing()
        num_rbf = self.smearing.num_rbf
        self.parameter_dimensions = (
            num_coefficients,
            num_coefficients,
            # num_rbf,
        )

        self.fc_node = torch.nn.Linear(
            2 * in_features,
            sum(self.parameter_dimensions),
        )

        self.fc_basis = torch.nn.Linear(
            num_rbf, num_coefficients, bias=False
        )

        self.fc_summary = torch.nn.Sequential(
            torch.nn.Linear(num_coefficients, num_coefficients),
            torch.nn.Tanh(),
            torch.nn.Linear(num_coefficients, out_features),
        )

        # self.fc_summary = torch.nn.Linear(num_coefficients, out_features)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
    ):  
        # (N, N, 2D)
        h_cat_ht = get_h_cat_ht(h)
        parameters = self.fc_node(h_cat_ht)
        K, Q = parameters.split(self.parameter_dimensions, dim=-1)

        # (N, N, 3)
        x_minus_xt = get_x_minus_xt(x)

        # (N, N, 1)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt)

        # (N, N, 3)
        x_minus_xt = x_minus_xt / (x_minus_xt_norm ** 2 + EPSILON)

        # (N, N, N_RBF)
        x_minus_xt_smear = self.smearing(x_minus_xt_norm)
        # x_minus_xt_smear = x_minus_xt_smear * C

        # (N, N, N_COEFFICIENTS)
        x_minus_xt_basis = self.fc_basis(x_minus_xt_smear)

        # (N, N, N_COEFFICIENTS, 3)
        x_minus_xt_basis = x_minus_xt_basis.unsqueeze(-1) * x_minus_xt.unsqueeze(-2)

        # (N, N, N_COEFFICIENTS, 3)
        x_k = x_minus_xt_basis * K.unsqueeze(-1)
        x_q = x_minus_xt_basis * Q.unsqueeze(-1)

        # (N, N, N_COEFFICIENTS)
        x_att = torch.einsum("...na, ...na -> ...n", x_k, x_q)

        # (N, N_COEFFICIENTS)
        h = x_att.sum(-2)

        # (N, OUT_FEATURES)
        h = self.fc_summary(h)
        return h

class GaussianDropout(torch.nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor):
        if self.training:
            x = x + torch.randn_like(x) * self.alpha
        else:
            return x










