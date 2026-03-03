"""Power iteration SCYFI — batched, no linear solve.

Forward-iterate z → f^k(z) until convergence. Each step is just
batched matmul + ReLU. Only finds stable cycles (|λ| < 1).

Uses the same minimal-iteration batched approach as VecC.
"""
# ruff: noqa: E501

from __future__ import annotations

import torch
from numpy import ndarray
from torch import Tensor

from dynamic.analysis.scyfi import _is_duplicate
from dynamic.analysis.scyfi_solve import fast_key


# ---------------------------------------------------------------------------
# Batched forward kernel
# ---------------------------------------------------------------------------
def _power_kernel(
    z: Tensor,
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
) -> tuple[Tensor, Tensor]:
    """One batched power step: z → f^k(z).

    Parameters
    ----------
    z : (B, dim)
    A, W : (dim, dim), h : (dim,)
    order : int

    Returns
    -------
    z_new : (B, dim) — f^k(z)
    residual : (B,) — ||f^k(z) - z||
    """
    z_orig = z.clone()
    for _ in range(order):
        d = (z > 0).to(z.dtype)
        AWD = A.unsqueeze(0) + W.unsqueeze(0) * d.unsqueeze(1)
        z = torch.bmm(AWD, z.unsqueeze(-1)).squeeze(-1) + h
    return z, (z - z_orig).norm(dim=-1)


# ---------------------------------------------------------------------------
# Core power iteration
# ---------------------------------------------------------------------------
def scy_fi_power(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    B: int = 256,
    outer_iterations: int = 3,
    inner_iterations: int = 20,
    tol: float = 1e-10,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Find k-cycles via batched power iteration.

    Matches VecC's iteration budget: outer × inner kernel calls
    with B candidates per call. Converged candidates are harvested
    and resampled; non-converged continue iterating.

    Only finds **stable** cycles (|λ| < 1).
    """
    dim = A.shape[0]
    dtype = A.dtype
    eye = torch.eye(dim, dtype=dtype)

    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    _scale = 1000
    found_keys: set[bytes] = set()
    for order_cycles in found_lower_orders:
        for t in order_cycles:
            for pt in t:
                found_keys.add(fast_key(pt, _scale))

    i = -1
    while i < outer_iterations:
        i += 1
        z = torch.randn(B, dim, dtype=dtype) * 2.0

        for _c in range(inner_iterations):
            z_new, residual = _power_kernel(z, A, W, h, order)
            converged = residual < tol

            # Harvest converged
            c_indices = converged.nonzero(as_tuple=False).squeeze(-1)
            for ci in c_indices:
                b = ci.item()
                z_pt = z_new[b]
                key0 = fast_key(z_pt, _scale)
                if key0 in found_keys:
                    z_new[b] = torch.randn(dim, dtype=dtype) * 2.0
                    continue

                # Build trajectory
                trajectory = []
                z_traj = z_pt.clone()
                for _ in range(order):
                    trajectory.append(z_traj.clone())
                    d = (z_traj > 0).to(dtype)
                    z_traj = (A + W * d) @ z_traj + h

                if not _is_duplicate(trajectory, cycles_found):
                    zf = eye.clone()
                    for k in range(order):
                        d_k = (trajectory[k] > 0).to(dtype)
                        zf = (A + W * d_k) @ zf
                    eigvals = torch.linalg.eigvals(zf).numpy()

                    cycles_found.append(trajectory)
                    eigvals_found.append(eigvals)
                    for pt in trajectory:
                        found_keys.add(fast_key(pt, _scale))
                    i = 0

                z_new[b] = torch.randn(dim, dtype=dtype) * 2.0

            z = z_new  # Continue iterating all B

    return cycles_found, eigvals_found


def find_cycles_power(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int,
    *,
    B: int = 256,
    outer_iterations: int = 3,
    inner_iterations: int = 20,
    tol: float = 1e-10,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find cycles via batched power iteration up to ``max_order``."""
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi_power(
            A,
            W,
            h,
            order,
            all_cycles,
            B=B,
            outer_iterations=outer_iterations,
            inner_iterations=inner_iterations,
            tol=tol,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals
