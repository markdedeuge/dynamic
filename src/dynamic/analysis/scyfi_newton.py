"""Newton-Raphson SCYFI — batched quadratic convergence.

Solves F(z) = f^k(z) - z = 0 via Newton's method. The Jacobian
of f^k is the chain product ∏ M_{D_i}. Finds both stable and
unstable cycles.

Uses the same minimal-iteration batched approach as VecC:
outer × inner kernel calls, B candidates per call.
"""
# ruff: noqa: E501

from __future__ import annotations

import torch
from numpy import ndarray
from torch import Tensor

from dynamic.analysis.scyfi import _is_duplicate
from dynamic.analysis.scyfi_solve import batched_solve, fast_key


# ---------------------------------------------------------------------------
# Batched Newton kernel
# ---------------------------------------------------------------------------
def _newton_kernel(
    z: Tensor,
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    eye: Tensor,
) -> tuple[Tensor, Tensor]:
    """One batched Newton step for all B candidates.

    Computes f^k(z), its Jacobian, and the Newton update.

    Parameters
    ----------
    z : (B, dim)
    A, W : (dim, dim), h : (dim,), eye : (dim, dim)
    order : int

    Returns
    -------
    z_new : (B, dim) — z - (J-I)^{-1}(f^k(z)-z)
    residual_norm : (B,) — ||f^k(z) - z||
    """
    B = z.shape[0]
    jacobian = eye.unsqueeze(0).expand(B, -1, -1).clone()
    z_cur = z.clone()

    for _ in range(order):
        d = (z_cur > 0).to(z.dtype)
        M = A.unsqueeze(0) + W.unsqueeze(0) * d.unsqueeze(1)
        jacobian = torch.bmm(M, jacobian)
        z_cur = torch.bmm(M, z_cur.unsqueeze(-1)).squeeze(-1) + h

    residual = z_cur - z  # f^k(z) - z
    J_minus_I = jacobian - eye.unsqueeze(0)

    delta = batched_solve(J_minus_I, residual.unsqueeze(-1)).squeeze(-1)
    nan_mask = torch.isnan(delta[:, 0])
    delta[nan_mask] = 0.0

    z_new = z - delta
    return z_new, residual.norm(dim=-1)


# ---------------------------------------------------------------------------
# Core Newton iteration
# ---------------------------------------------------------------------------
def scy_fi_newton(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    B: int = 64,
    outer_iterations: int = 2,
    inner_iterations: int = 5,
    tol: float = 1e-12,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Find k-cycles via batched Newton-Raphson.

    Matches VecC's iteration budget: outer × inner kernel calls
    with B candidates per call. Converged candidates are harvested
    and resampled.

    Finds both stable and unstable cycles.
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
            z_new, res_norm = _newton_kernel(z, A, W, h, order, eye)
            converged = res_norm < tol

            # Harvest converged
            c_indices = converged.nonzero(as_tuple=False).squeeze(-1)
            for ci in c_indices:
                b = ci.item()
                z_pt = z_new[b]
                key0 = fast_key(z_pt, _scale)
                if key0 in found_keys:
                    z_new[b] = torch.randn(dim, dtype=dtype) * 2.0
                    continue

                # Build + verify trajectory
                trajectory = []
                z_traj = z_pt.clone()
                for _ in range(order):
                    trajectory.append(z_traj.clone())
                    d = (z_traj > 0).to(dtype)
                    z_traj = (A + W * d) @ z_traj + h

                if not torch.allclose(z_traj, z_pt, atol=tol * 100):
                    z_new[b] = torch.randn(dim, dtype=dtype) * 2.0
                    continue

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

            z = z_new

            # Break if all candidates converged in this iteration
            if converged.all():
                break

    return cycles_found, eigvals_found


def find_cycles_newton(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int,
    *,
    B: int = 64,
    outer_iterations: int = 2,
    inner_iterations: int = 5,
    tol: float = 1e-12,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find cycles via batched Newton-Raphson up to ``max_order``."""
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi_newton(
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
