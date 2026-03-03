"""Hybrid SCYFI — D-vector iteration + Newton polish.

Phase 1 (D-vec Iteration): sample random D-vector sequences, solve
    the linear system, observe actual D-vectors, feed back as next
    proposal. Iterates until D-vectors converge (like VecC inner loop).
Phase 2 (Newton Polish): for near-consistent solutions (where the
    observed D-vector almost matches), refine with Newton-Raphson.

Key advantage over original power-iteration Hybrid:
- Finds both stable and unstable cycles (not just attractors)
- D-vector iteration converges much faster than point iteration
- Newton phase captures boundary cycles that VecC misses
"""
# ruff: noqa: E501

from __future__ import annotations

import torch
from numpy import ndarray
from torch import Tensor

from dynamic.analysis.scyfi import _is_duplicate
from dynamic.analysis.scyfi_solve import batched_solve, fast_key


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _dvec_solve_and_observe(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    d_vecs: Tensor,
    order: int,
    eye: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Solve for fixed points given D-vecs, return observed D-vecs.

    Parameters
    ----------
    d_vecs : (B, order, dim) — proposed D-vector sequences

    Returns
    -------
    z_cands : (B, dim) — solved candidates
    traj : (B, order, dim) — simulated trajectories
    traj_d : (B, order, dim) — observed D-vectors
    consistent : (B,) — proposed == observed
    """
    B = d_vecs.shape[0]
    dim = A.shape[0]

    # Chain product: z_factor = Π(A + W * d[k])
    z_factors = eye.unsqueeze(0).expand(B, -1, -1).clone()
    h_factors = z_factors.clone()
    for k in range(order):
        AWDs = A.unsqueeze(0) + W.unsqueeze(0) * d_vecs[:, k, :].unsqueeze(1)
        z_factors = torch.bmm(AWDs, z_factors)
        if k > 0:
            h_factors = torch.bmm(AWDs, h_factors) + eye.unsqueeze(0)

    # Solve: z = (I - z_factor)^{-1} @ (h_factor @ h)
    lhs = eye.unsqueeze(0) - z_factors
    rhs = (h_factors @ h).unsqueeze(-1)
    z_cands = batched_solve(lhs, rhs).squeeze(-1)

    # Simulate to get observed D-vectors
    traj = torch.empty(B, order, dim, dtype=A.dtype)
    traj_d = torch.empty(B, order, dim, dtype=A.dtype)
    z = z_cands
    for k in range(order):
        traj[:, k] = z
        d = (z > 0).to(A.dtype)
        traj_d[:, k] = d
        AWD_sim = A.unsqueeze(0) + W.unsqueeze(0) * d.unsqueeze(1)
        z = torch.bmm(AWD_sim, z.unsqueeze(-1)).squeeze(-1) + h

    # Consistency: proposed == observed
    consistent = (d_vecs == traj_d).all(dim=-1).all(dim=-1)
    nan_mask = torch.isnan(z_cands[:, 0])
    consistent = consistent & ~nan_mask

    return z_cands, traj, traj_d, consistent


def _newton_step(
    z: Tensor,
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    eye: Tensor,
) -> Tensor:
    """One batched Newton update: z - (J-I)^{-1}(f^k(z)-z)."""
    B = z.shape[0]
    jacobian = eye.unsqueeze(0).expand(B, -1, -1).clone()
    z_cur = z.clone()

    for _ in range(order):
        d = (z_cur > 0).to(z.dtype)
        M = A.unsqueeze(0) + W.unsqueeze(0) * d.unsqueeze(1)
        jacobian = torch.bmm(M, jacobian)
        z_cur = torch.bmm(M, z_cur.unsqueeze(-1)).squeeze(-1) + h

    residual = z_cur - z
    J_minus_I = jacobian - eye.unsqueeze(0)
    delta = batched_solve(J_minus_I, residual.unsqueeze(-1)).squeeze(-1)
    nan_mask = torch.isnan(delta[:, 0])
    delta[nan_mask] = 0.0
    return z - delta


# ---------------------------------------------------------------------------
# Harvest utility
# ---------------------------------------------------------------------------
def _harvest(
    z: Tensor,
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    tol: float,
    cycles_found: list[list[Tensor]],
    eigvals_found: list[ndarray],
    found_keys: set[bytes],
    scale: int = 1000,
) -> bool:
    """Check converged candidates and harvest unique cycles."""
    dim = A.shape[0]
    dtype = A.dtype
    eye = torch.eye(dim, dtype=dtype)
    found_new = False

    z_check = z.clone()
    for _ in range(order):
        d = (z_check > 0).to(dtype)
        z_check = (
            A.unsqueeze(0) + W.unsqueeze(0) * d.unsqueeze(1)
        ) @ z_check.unsqueeze(-1)
        z_check = z_check.squeeze(-1) + h

    converged = (z_check - z).norm(dim=-1) < tol
    c_indices = converged.nonzero(as_tuple=False).squeeze(-1)

    for ci in c_indices:
        b = ci.item()
        z_pt = z[b]
        key0 = fast_key(z_pt, scale)
        if key0 in found_keys:
            continue

        trajectory = []
        z_traj = z_pt.clone()
        for _ in range(order):
            trajectory.append(z_traj.clone())
            d = (z_traj > 0).to(dtype)
            z_traj = (A + W * d) @ z_traj + h

        if not torch.allclose(z_traj, z_pt, atol=1e-6):
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
                found_keys.add(fast_key(pt, scale))
            found_new = True

    return found_new


# ---------------------------------------------------------------------------
# Main: D-vector iteration + Newton polish
# ---------------------------------------------------------------------------
def scy_fi_hybrid(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    B: int = 256,
    outer_iterations: int = 2,
    inner_iterations: int = 3,
    newton_steps: int = 3,
    max_mismatches: int = 2,
    tol: float = 1e-10,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Two-phase hybrid: D-vector iteration + Newton polish.

    Phase 1 (per outer iteration):
        - Sample B random D-vector sequences
        - For ``inner_iterations`` rounds:
            solve → observe → feed observed D-vecs as next proposal
        - Harvest any self-consistent solutions

    Phase 2:
        - For near-consistent solutions (≤ max_mismatches),
          apply Newton refinement to find boundary cycles.

    Parameters
    ----------
    B : int
        D-vector samples per outer iteration.
    outer_iterations : int
        Number of resample rounds (resets on new cycle found).
    inner_iterations : int
        D-vector feedback iterations (solve → observe → re-propose).
    newton_steps : int
        Newton refinement steps on near-consistent candidates.
    max_mismatches : int
        Maximum D-vector mismatches for Newton phase.
    tol : float
        Convergence tolerance for cycle validation.
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

        # Start with random D-vectors
        d_vecs = (torch.rand(B, order, dim, dtype=dtype) > 0.5).to(dtype)

        # Phase 1: D-vector iteration (solve → observe → re-propose)
        for _ in range(inner_iterations):
            z_cands, traj, traj_d, consistent = _dvec_solve_and_observe(
                A,
                W,
                h,
                d_vecs,
                order,
                eye,
            )

            # Harvest any exact matches found during iteration
            c_indices = consistent.nonzero(as_tuple=False).squeeze(-1)
            if c_indices.numel() > 0:
                z_exact = z_cands[consistent]
                if _harvest(
                    z_exact,
                    A,
                    W,
                    h,
                    order,
                    tol,
                    cycles_found,
                    eigvals_found,
                    found_keys,
                    _scale,
                ):
                    i = 0

            # Feed observed D-vectors back as next proposal
            all_converged = (d_vecs == traj_d).all()
            d_vecs = traj_d.clone()

            # Break inner loop if all D-vecs were already stable
            if all_converged:
                break

        # Phase 2: Newton polish on near-consistent candidates
        # (those where D-vec almost matches after iteration)
        mismatches = (d_vecs != traj_d).sum(dim=-1).sum(dim=-1)
        near = (mismatches > 0) & (mismatches <= max_mismatches)
        near = near & ~torch.isnan(z_cands[:, 0])

        if near.any():
            z_near = z_cands[near]
            for _ in range(newton_steps):
                z_near = _newton_step(z_near, A, W, h, order, eye)

            if _harvest(
                z_near,
                A,
                W,
                h,
                order,
                tol,
                cycles_found,
                eigvals_found,
                found_keys,
                _scale,
            ):
                i = 0

    return cycles_found, eigvals_found


def find_cycles_hybrid(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int,
    *,
    B: int = 256,
    outer_iterations: int = 2,
    inner_iterations: int = 3,
    newton_steps: int = 3,
    max_mismatches: int = 2,
    tol: float = 1e-10,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Hybrid D-vector iteration + Newton cycle finder."""
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi_hybrid(
            A,
            W,
            h,
            order,
            all_cycles,
            B=B,
            outer_iterations=outer_iterations,
            inner_iterations=inner_iterations,
            newton_steps=newton_steps,
            max_mismatches=max_mismatches,
            tol=tol,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals
