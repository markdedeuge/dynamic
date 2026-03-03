"""Exhaustive SCYFI — enumerate ALL D-vector sequences.

For dim=D and order=k, there are (2^D)^k possible D-vector sequences.
At small dim×k, we enumerate ALL of them, batch-solve all linear systems,
and check consistency in one shot — guaranteed to find ALL cycles.

Feasible for 2^(dim*order) ≤ ~100K on CPU.
"""
# ruff: noqa: E501

from __future__ import annotations

import torch
from numpy import ndarray
from torch import Tensor

from dynamic.analysis.scyfi import _is_duplicate
from dynamic.analysis.scyfi_solve import batched_solve, fast_key


# ---------------------------------------------------------------------------
# D-vector sequence generation
# ---------------------------------------------------------------------------
def _generate_all_d_sequences(dim: int, order: int, dtype: torch.dtype) -> Tensor:
    """Generate all (2^dim)^order D-vector sequences.

    Returns shape ``(N, order, dim)`` where ``N = (2^dim)^order``.
    """
    n_single = 1 << dim  # 2^dim
    # All single-step D-vectors: (2^dim, dim)
    indices = torch.arange(n_single, dtype=torch.long)
    d_single = torch.zeros(n_single, dim, dtype=dtype)
    for bit in range(dim):
        d_single[:, bit] = ((indices >> bit) & 1).to(dtype)

    if order == 1:
        return d_single.unsqueeze(1)  # (N, 1, dim)

    # Cartesian product via meshgrid indices
    grids = torch.meshgrid(
        *[torch.arange(n_single) for _ in range(order)],
        indexing="ij",
    )
    # Stack and reshape: (N, order) indices into d_single
    idx = torch.stack(grids, dim=-1).reshape(-1, order)  # (N, order)
    return d_single[idx]  # (N, order, dim)


# ---------------------------------------------------------------------------
# Core exhaustive search
# ---------------------------------------------------------------------------
def scy_fi_exhaustive(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    max_systems: int = 100_000,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Find ALL k-cycles by exhaustive enumeration.

    Parameters
    ----------
    A, W, h : Tensor
        PLRNN parameters.
    order : int
        Cycle order.
    found_lower_orders : list
        Previously found cycles.
    max_systems : int
        Max number of systems to enumerate. Raises ValueError if exceeded.

    Returns
    -------
    tuple[list, list]
        ``(cycles_found, eigvals_found)``.
    """
    dim = A.shape[0]
    dtype = A.dtype
    n_systems = (1 << dim) ** order

    if n_systems > max_systems:
        raise ValueError(
            f"Exhaustive search requires {n_systems} systems "
            f"(dim={dim}, order={order}), exceeds max_systems={max_systems}. "
            f"Use a heuristic method instead."
        )

    eye = torch.eye(dim, dtype=dtype)

    # Generate ALL D-vector sequences
    d_all = _generate_all_d_sequences(dim, order, dtype)  # (N, order, dim)
    N = d_all.shape[0]

    # Batched chain product
    z_factors = eye.unsqueeze(0).expand(N, -1, -1).clone()
    h_factors = z_factors.clone()
    for k in range(order):
        AWDs = A.unsqueeze(0) + W.unsqueeze(0) * d_all[:, k, :].unsqueeze(1)
        z_factors = torch.bmm(AWDs, z_factors)
        if k > 0:
            h_factors = torch.bmm(AWDs, h_factors) + eye.unsqueeze(0)

    # Batched solve
    lhs = eye.unsqueeze(0) - z_factors
    rhs = (h_factors @ h).unsqueeze(-1)
    z_cands = batched_solve(lhs, rhs).squeeze(-1)  # (N, dim)

    # Batched simulation + consistency check
    traj = torch.empty(N, order, dim, dtype=dtype)
    traj_d = torch.empty(N, order, dim, dtype=dtype)
    z = z_cands
    for k in range(order):
        traj[:, k] = z
        d = (z > 0).to(dtype)
        traj_d[:, k] = d
        AWD_sim = A.unsqueeze(0) + W.unsqueeze(0) * d.unsqueeze(1)
        z = torch.bmm(AWD_sim, z.unsqueeze(-1)).squeeze(-1) + h

    # Consistency: proposed D == observed D, and no NaN
    consistent = (d_all == traj_d).all(dim=-1).all(dim=-1)
    nan_mask = torch.isnan(z_cands[:, 0])
    consistent = consistent & ~nan_mask

    # Harvest unique cycles
    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    _scale = 1000
    found_keys: set[bytes] = set()
    for order_cycles in found_lower_orders:
        for t in order_cycles:
            for pt in t:
                found_keys.add(fast_key(pt, _scale))

    valid_indices = consistent.nonzero(as_tuple=False).squeeze(-1)
    for vi in valid_indices:
        i = vi.item()
        key0 = fast_key(traj[i, 0], _scale)
        if key0 in found_keys:
            continue

        trajectory = [traj[i, j].clone() for j in range(order)]
        if not _is_duplicate(trajectory, cycles_found):
            zf = eye.clone()
            for k in range(order):
                zf = (A + W * d_all[i, k]) @ zf
            eigvals = torch.linalg.eigvals(zf).numpy()

            cycles_found.append(trajectory)
            eigvals_found.append(eigvals)
            for pt in trajectory:
                found_keys.add(fast_key(pt, _scale))

    return cycles_found, eigvals_found


def find_cycles_exhaustive(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int,
    *,
    max_systems: int = 100_000,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find all cycles up to ``max_order`` by exhaustive enumeration.

    Guaranteed complete for small dim. Raises ValueError if
    the number of systems exceeds ``max_systems``.
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi_exhaustive(
            A,
            W,
            h,
            order,
            all_cycles,
            max_systems=max_systems,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals
