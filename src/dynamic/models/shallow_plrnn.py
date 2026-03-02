"""Shallow PLRNN (shPLRNN).

z_t = diag(A) z_{t-1} + W1 ReLU(W2 z_{t-1} + h2) + h1

With W1 ∈ R^{M×H}, W2 ∈ R^{H×M}, the hidden layer introduces
up to ∑_{k=0}^{M} C(H,k) linear subregions.

Reference: §3.1, Hess et al. (2023).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ShallowPLRNN(nn.Module):
    """Shallow PLRNN with a single hidden layer.

    Parameters
    ----------
    M : int
        State-space dimensionality.
    H : int
        Hidden layer dimensionality.
    """

    def __init__(self, M: int, H: int) -> None:
        super().__init__()
        self.M = M
        self.H = H
        self.A = nn.Parameter(torch.randn(M) * 0.1 + 0.9)
        self.W1 = nn.Parameter(torch.randn(M, H) * 0.1)
        self.W2 = nn.Parameter(torch.randn(H, M) * 0.1)
        self.h1 = nn.Parameter(torch.zeros(M))
        self.h2 = nn.Parameter(torch.zeros(H))

    def forward(self, z: Tensor) -> Tensor:
        """Compute z_t = diag(A) z_{t-1} + W1 ReLU(W2 z_{t-1} + h2) + h1.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)`` or ``(B, M)``.

        Returns
        -------
        Tensor
            Next state, same shape as input.
        """
        hidden = z @ self.W2.T + self.h2
        return self.A * z + hidden.clamp(min=0.0) @ self.W1.T + self.h1

    def get_D_hidden(self, z: Tensor) -> Tensor:
        """Diagonal activation matrix for the hidden layer.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)``.

        Returns
        -------
        Tensor
            Diagonal matrix of shape ``(H, H)`` with ``D[i,i] = 1`` if
            ``(W2 z + h2)[i] > 0``, else ``0``.
        """
        hidden = self.W2 @ z + self.h2
        return torch.diag((hidden > 0).float())

    def get_jacobian(self, z: Tensor) -> Tensor:
        """Jacobian: diag(A) + W1 @ D_h @ W2.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)``.

        Returns
        -------
        Tensor
            Jacobian matrix of shape ``(M, M)``.
        """
        D_h = self.get_D_hidden(z)
        return torch.diag(self.A) + self.W1 @ D_h @ self.W2

    def get_subregion_id(self, z: Tensor) -> tuple[int, ...]:
        """Subregion ID based on hidden-layer activation pattern.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)``.

        Returns
        -------
        tuple[int, ...]
            Binary tuple of length ``H``.
        """
        hidden = self.W2 @ z + self.h2
        return tuple(int(x) for x in (hidden > 0).tolist())

    @torch.no_grad()
    def forward_trajectory(self, z0: Tensor, T: int) -> Tensor:
        """Generate a trajectory of length T starting from z0.

        Parameters
        ----------
        z0 : Tensor
            Initial state of shape ``(M,)``.
        T : int
            Number of time steps.

        Returns
        -------
        Tensor
            Trajectory of shape ``(T + 1, M)``.
        """
        states = [z0]
        z = z0
        for _ in range(T):
            z = self.forward(z)
            states.append(z)
        return torch.stack(states)
