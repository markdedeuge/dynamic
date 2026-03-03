"""Almost-Linear RNN (ALRNN).

Only the last P dimensions use ReLU; the first M-P are purely linear.
This gives 2^P subregions instead of 2^M.

    Φ(z_t) = [z_1, ..., z_{M-P}, max(0, z_{M-P+1}), ..., max(0, z_M)]

Reference: §3.1, Brenner et al. (2024a).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ALRNN(nn.Module):
    """Almost-Linear Recurrent Neural Network.

    Parameters
    ----------
    M : int
        State-space dimensionality.
    P : int
        Number of ReLU dimensions (last P of M).
    """

    def __init__(self, M: int, P: int) -> None:
        super().__init__()
        if P > M:
            msg = f"P ({P}) must be <= M ({M})"
            raise ValueError(msg)
        self.M = M
        self.P = P
        self.A = nn.Parameter(torch.randn(M) * 0.1 + 0.9)
        self.W = nn.Parameter(torch.randn(M, M) * 0.1)
        self.h = nn.Parameter(torch.zeros(M))

    def forward(self, z: Tensor) -> Tensor:
        """Compute z_t = diag(A) z_{t-1} + W Φ(z_{t-1}) + h.

        The activation Φ applies identity to the first M-P dimensions
        and ReLU to the last P dimensions.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)`` or ``(B, M)``.

        Returns
        -------
        Tensor
            Next state, same shape as input.
        """
        linear_part = z[..., : self.M - self.P]
        relu_part = z[..., -self.P :].clamp(min=0.0)
        phi = torch.cat([linear_part, relu_part], dim=-1)
        return self.A * z + phi @ self.W.T + self.h

    def get_D(self, z: Tensor) -> Tensor:
        """Diagonal activation matrix: first M-P always 1.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)``.

        Returns
        -------
        Tensor
            Diagonal matrix of shape ``(M, M)``.
        """
        d = torch.ones(self.M)
        d[-self.P :] = (z[-self.P :] > 0).float()
        return torch.diag(d)

    def get_jacobian(self, z: Tensor) -> Tensor:
        """Jacobian: diag(A) + W @ D(z).

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
        """Subregion ID from activation pattern.

        First M-P entries are always 1. Only last P entries vary.

        Parameters
        ----------
        z : Tensor
            State vector of shape ``(M,)``.

        Returns
        -------
        tuple[int, ...]
            Binary tuple of length ``M``.
        """
        d = [1] * (self.M - self.P)
        d.extend(int(x) for x in (z[-self.P :] > 0).tolist())
        return tuple(d)

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
