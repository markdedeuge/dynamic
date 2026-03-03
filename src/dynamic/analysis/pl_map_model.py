"""nn.Module wrapper for PL maps with full A matrix.

The PL map's PLRNN reformulation yields a full 2×2 matrix A (not
diagonal), which cannot be represented by the standard diagonal-A
PLRNN class. This wrapper preserves the (A + WD)z + h structure
and provides the standard API: forward, get_jacobian, get_D,
get_subregion_id, forward_trajectory.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dynamic.systems.pl_map import PLMap


class PLMapModel(nn.Module):
    """nn.Module wrapper around PL map ``(A + WD) z + h``.

    Attributes
    ----------
    A : Parameter
        Full state transition matrix, shape ``(M, M)``.
    W : Parameter
        Activation weight matrix, shape ``(M, M)``.
    h : Parameter
        Bias vector, shape ``(M,)``.
    M : int
        State dimension.
    """

    def __init__(self, pl_map: PLMap):
        super().__init__()
        params = pl_map.to_plrnn_params()
        self.A = nn.Parameter(
            torch.as_tensor(params["A"], dtype=torch.float32),
        )
        self.W = nn.Parameter(
            torch.as_tensor(params["W"], dtype=torch.float32),
        )
        self.h = nn.Parameter(
            torch.as_tensor(params["h"], dtype=torch.float32),
        )
        self.M = self.h.shape[0]

    def forward(self, z: Tensor) -> Tensor:
        """Forward step: z_t = (A + W D(z)) z + h."""
        D = torch.diag((z > 0).float())
        return (self.A + self.W @ D) @ z + self.h

    def get_D(self, z: Tensor) -> Tensor:
        """Activation matrix D = diag(z > 0)."""
        return torch.diag((z > 0).float())

    def get_jacobian(self, z: Tensor) -> Tensor:
        """Jacobian J = A + W D(z)."""
        return self.A + self.W @ self.get_D(z)

    def get_subregion_id(self, z: Tensor) -> tuple:
        """Binary sign pattern of z."""
        return tuple(int(x > 0) for x in z.tolist())

    def forward_trajectory(self, z0: Tensor, T: int) -> Tensor:
        """Generate forward trajectory of length T.

        Returns tensor of shape ``(T + 1, M)``.
        """
        traj = [z0]
        z = z0
        with torch.no_grad():
            for _ in range(T):
                z = self.forward(z)
                traj.append(z)
        return torch.stack(traj)
