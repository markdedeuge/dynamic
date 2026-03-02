"""Lyapunov exponent computation via QR decomposition.

Computes the full Lyapunov spectrum of a PLRNN by accumulating
Jacobian products along a trajectory and using the QR decomposition
method (Benettin et al., 1980).

Reference: Standard method for computing Lyapunov exponents.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor, nn


def compute_lyapunov_exponents(
    model: nn.Module,
    z0: Tensor,
    T: int = 10000,
    transient: int = 100,
) -> ndarray:
    """Compute Lyapunov spectrum via QR decomposition of Jacobian products.

    Uses the standard algorithm:
    1. Evolve trajectory z_0, z_1, ..., z_T
    2. At each step, multiply Q by J(z_t) and QR-decompose
    3. Accumulate log(|R_ii|) for each exponent

    Parameters
    ----------
    model : nn.Module
        PLRNN model with ``get_jacobian(z)`` method.
    z0 : Tensor
        Initial state of shape ``(M,)``.
    T : int
        Number of iterations for the computation.
    transient : int
        Number of initial steps to discard (warm-up).

    Returns
    -------
    ndarray
        Lyapunov exponents in descending order, shape ``(M,)``.
    """
    M = z0.shape[0]
    z = z0.clone().detach()

    # Initialize Q as identity
    Q = torch.eye(M, dtype=torch.float64)

    # Warm-up: discard transient
    with torch.no_grad():
        for _ in range(transient):
            z = model.forward(z)

    # Accumulate log stretching factors
    log_R = torch.zeros(M, dtype=torch.float64)

    with torch.no_grad():
        for _t in range(T):
            J = model.get_jacobian(z).double()
            z = model.forward(z)

            # Multiply: M = J @ Q
            M_mat = J @ Q

            # QR decomposition
            Q, R = torch.linalg.qr(M_mat)

            # Accumulate log of diagonal (absolute values)
            diag_R = torch.abs(torch.diag(R))
            # Guard against zeros
            diag_R = torch.clamp(diag_R, min=1e-300)
            log_R += torch.log(diag_R)

    # Average over time
    exponents = (log_R / T).numpy()

    # Sort descending
    exponents = np.sort(exponents)[::-1].copy()

    return exponents
