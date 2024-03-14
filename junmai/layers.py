from typing import Optional
import math

from requests import get
import torch
from .utils import (
    ExpNormalSmearing,
    get_h_cat_ht,
    get_x_minus_xt,
    get_x_minus_xt_norm,
    EPSILON,
)

class SakeLayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        pass

class JunmaiLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_features: Optional[int] = None,
        num_rbf: Optional[int] = None,
        out_features: int = 1,
        smearing: torch.nn.Module = ExpNormalSmearing,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.smearing = smearing(num_rbf=num_rbf)
        num_rbf = self.smearing.num_rbf
        self.num_rbf = num_rbf
        # self.fc_basis = torch.nn.Linear(
        #     num_rbf, hidden_features, # bias=False
        # )

        self.fc_summary = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_features, out_features),
        )

        # self.fc_coefficient = torch.nn.Linear(
        #     in_features, 4 * num_rbf * hidden_features
        # )

    def forward(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
    ):  
        K, Q = W

        # (N, N, 3)
        x_minus_xt = x.unsqueeze(-2) - x.unsqueeze(-3)

        # (N, N, 1)
        # x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt)
        x_minus_xt_norm_sq = (x_minus_xt.pow(2).sum(-1, keepdims=True) + EPSILON)
        x_minus_xt_norm = x_minus_xt_norm_sq.sqrt()

        # (N, N, 3)
        x_minus_xt = x_minus_xt / x_minus_xt_norm_sq

        # (N, N, N_RBF)
        x_minus_xt_smear = self.smearing(x_minus_xt_norm)

        # (N, N, N_RBF, 3)
        x_minus_xt_basis = x_minus_xt_smear.unsqueeze(-1) * x_minus_xt.unsqueeze(-2)

        # (N, N, N_COEFFICIENT, 3)
        x_minus_xt_basis_k = torch.einsum(
            "...nab, ...nac -> ...cb",
            x_minus_xt_basis,
            K,
        )

        x_minus_xt_basis_q = torch.einsum(
            "...nab, ...nac -> ...cb",
            x_minus_xt_basis,
            K,
        )

        # (N, N, N_COEFFICIENT)
        x_att = torch.einsum(
            "...ab, ...ab -> ...a",
            x_minus_xt_basis_k,
            x_minus_xt_basis_q,
        )

        # (N, N, N_COEFFICIENT)
        x_att = self.fc_summary(x_att).sum(-2)
        return x_att

class GaussianDropout(torch.nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor):
        if self.training:
            x = x + torch.randn_like(x) * self.alpha
        return x










