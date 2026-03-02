"""Loss functions for PLRNN training.

- MSE loss for trajectory reconstruction
- Invertibility regularization (Eq 5) to enforce map invertibility

Reference: §3.4, Eq 5.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error loss.

    Computes ``(1 / (M * T)) * sum ||pred_t - target_t||^2``.

    Parameters
    ----------
    predictions : Tensor
        Predicted states, shape ``(T, M)`` or ``(B, T, M)``.
    targets : Tensor
        Target states, same shape as predictions.

    Returns
    -------
    Tensor
        Scalar loss value.
    """
    return nn.functional.mse_loss(predictions, targets)


def invertibility_regularization(
    model: nn.Module,
    jacobians: list[Tensor],
    lambda_inv: float = 1.0,
) -> Tensor:
    """Invertibility regularization (Eq 5).

    Penalizes negative Jacobian determinants to enforce invertibility
    of the piecewise-linear map F_θ.

    L_reg = λ · (1/|S|) · Σ max(0, -det(J_i))

    Parameters
    ----------
    model : nn.Module
        PLRNN model (used for gradient graph attachment).
    jacobians : list[Tensor]
        Pre-computed Jacobian matrices for sampled subregions.
        Works with any model type (PLRNN, ShallowPLRNN, ALRNN).
    lambda_inv : float
        Regularization strength.

    Returns
    -------
    Tensor
        Scalar regularization loss.
    """
    if not jacobians or lambda_inv == 0.0:
        return torch.tensor(0.0, requires_grad=True)

    penalty = torch.tensor(0.0)
    for J in jacobians:
        det = torch.det(J)
        penalty = penalty + torch.clamp(-det, min=0.0)

    return lambda_inv * penalty / len(jacobians)
