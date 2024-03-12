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
        # # (N, N_COEFFICIENTS, 4 * N_COEFFICIENTS)
        # h = self.fc_coefficient(h)
        # h = h.reshape(*h.shape[:-1], self.num_rbf, 4 * self.hidden_features)

        # # (N, N, N, N_COEFFICIENTS, N_COEFFICIENTS)
        # h0, h1, h2, h3 = h.chunk(4, dim=-1)
        # h = torch.einsum("...iab, ...jab, ...kab, ...lab -> ...ijklab", h0, h1, h2, h3)

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

        # (N, N, N_COEFFICIENTS)
        # x_minus_xt_basis = self.fc_basis(x_minus_xt_smear)
        x_minus_xt_basis = x_minus_xt_smear

        # (N, N, N_COEFFICIENTS, 3)
        x_minus_xt_basis = x_minus_xt_basis.unsqueeze(-1) * x_minus_xt.unsqueeze(-2)

        # (N_COEFFICIENTS,)
        h = torch.einsum(
            "...abkc, ...dekc, ...abdeko -> ...ao",
            x_minus_xt_basis,
            x_minus_xt_basis,
            W,
        )
        h = self.fc_summary(h)

        return h
    
    # def forward(self, h, x):
    #     delta_x = x.unsqueeze(-2) - x.unsqueeze(-3)
    #     delta_x_norm = delta_x.pow(2).sum(-1, keepdims=True)
    #     delta_x_norm_smeared = self.smearing(delta_x_norm)
    #     delta_x_norm_smeared = self.fc_basis(delta_x_norm_smeared)
    #     delta_x = delta_x / (delta_x_norm + EPSILON)
    #     delta_x = delta_x.unsqueeze(-2) * delta_x_norm_smeared.unsqueeze(-1)
    #     delta_x = torch.einsum("...abkc, ...dekc, abdeko -> ...o", delta_x, delta_x, self.W)
    #     return self.fc_summary(delta_x)


class GaussianDropout(torch.nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor):
        if self.training:
            x = x + torch.randn_like(x) * self.alpha
        return x










