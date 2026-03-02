"""Base PLRNN: z_t = diag(A) z_{t-1} + W ReLU(z_{t-1}) + h.

Piecewise-linear form (Eq 2):
    z_t = (diag(A) + W D_{t-1}) z_{t-1} + h
where D_t = diag(z_t > 0).

Reference: §3.1, Eq 1-2.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PLRNN(nn.Module):
    """Piecewise-Linear Recurrent Neural Network.

    Parameters
    ----------
    M : int
        State-space dimensionality.
    """

    def __init__(self, M: int) -> None:
        super().__init__()
        self.M = M
        self.A = nn.Parameter(torch.randn(M) * 0.1 + 0.9)
        self.W = nn.Parameter(torch.randn(M, M) * 0.1)
        self.h = nn.Parameter(torch.zeros(M))

    def forward(self, z: Tensor) -> Tensor:
        """Compute z_t = diag(A) * z_{t-1} + W @ ReLU(z_{t-1}) + h.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)`` or ``(B, M)``.

        Returns
        -------
        Tensor
            Next state, same shape as input.
        """
        return self.A * z + z.clamp(min=0.0) @ self.W.T + self.h

    def get_D(self, z: Tensor) -> Tensor:
        """Diagonal ReLU activation matrix.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)``.

        Returns
        -------
        Tensor
            Diagonal matrix of shape ``(M, M)`` with ``D[i,i] = 1`` if
            ``z[i] > 0``, else ``0``.
        """
        return torch.diag((z > 0).float())

    def get_jacobian(self, z: Tensor) -> Tensor:
        """Jacobian of the map: diag(A) + W @ D(z).

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)``.

        Returns
        -------
        Tensor
            Jacobian matrix of shape ``(M, M)``.
        """
        D = self.get_D(z)
        return torch.diag(self.A) + self.W @ D

    def get_subregion_id(self, z: Tensor) -> tuple[int, ...]:
        """Hashable subregion identifier from sign pattern.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)``.

        Returns
        -------
        tuple[int, ...]
            Binary tuple of length ``M``.
        """
        return tuple(int(x) for x in (z > 0).tolist())

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
