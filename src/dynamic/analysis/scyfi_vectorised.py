"""Vectorised SCYFI v2 — minimal Python overhead.

Processes B candidates through chain-product → solve → simulate
WITHOUT per-member active-set tracking. All B members processed
unconditionally every iteration. Harvesting is deferred to
post-iteration with cheap tensor comparisons.

Uses extracted kernel functions for the hot tensor ops.
"""
# ruff: noqa: E501

from __future__ import annotations

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from dynamic.analysis.scyfi import _is_duplicate
from dynamic.analysis.scyfi_helpers import (
    construct_relu_matrix_pool,
    set_loop_iterations,
)
from dynamic.analysis.scyfi_solve import (
    auto_batch_size,
    batched_solve,
    build_awd_table,
    d_vecs_to_indices,
    fast_key,
)


# ---------------------------------------------------------------------------
# Kernel: batched chain product + solve + simulate + check in one call
# ---------------------------------------------------------------------------
def _full_iteration_kernel(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    d_vecs: Tensor,
    order: int,
    eye: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One complete inner iteration for all B candidates.

    Parameters
    ----------
    A : (dim, dim), W : (dim, dim), h : (dim,)
    d_vecs : (B, order, dim) — proposed D-vectors
    order : int
    eye : (dim, dim)

    Returns
    -------
    z_factors : (B, dim, dim) — chain product result
    traj : (B, order, dim) — simulated trajectory
    traj_d : (B, order, dim) — observed D-vectors from trajectory
    consistent : (B,) — whether proposed == observed D-vectors
    """
    B = d_vecs.shape[0]

    # Chain product: z_factor = Π(A + W * d[k])
    z_factors = eye.unsqueeze(0).expand(B, -1, -1).clone()
    h_factors = z_factors.clone()
    for k in range(order):
        AWDs = A.unsqueeze(0) + W.unsqueeze(0) * d_vecs[:, k, :].unsqueeze(1)
        z_factors = torch.bmm(AWDs, z_factors)
        if k > 0:
            h_factors = torch.bmm(AWDs, h_factors) + eye.unsqueeze(0)

    # Solve: z_cand = (I - z_factor)^{-1} @ (h_factor @ h)
    lhs = eye.unsqueeze(0) - z_factors
    rhs = (h_factors @ h).unsqueeze(-1)

    # Use solve with error handling — singular matrices get NaN
    try:
        z_cands = torch.linalg.solve(lhs, rhs).squeeze(-1)
    except torch.linalg.LinAlgError:
        # Batched solve failed — fall back to per-item
        dim = A.shape[0]
        z_cands = torch.full((B, dim), float("nan"), dtype=A.dtype)
        for b in range(B):
            try:
                z_cands[b] = torch.linalg.solve(lhs[b], rhs[b]).squeeze(-1)
            except torch.linalg.LinAlgError:
                pass  # remains NaN

    # Simulate trajectory + extract D-vecs
    dim = A.shape[0]
    traj = torch.empty(B, order, dim, dtype=A.dtype)
    traj_d = torch.empty(B, order, dim, dtype=A.dtype)
    z = z_cands
    for k in range(order):
        traj[:, k] = z
        d = (z > 0).to(A.dtype)
        traj_d[:, k] = d
        AWD_sim = A.unsqueeze(0) + W.unsqueeze(0) * d.unsqueeze(1)
        z = torch.bmm(AWD_sim, z.unsqueeze(-1)).squeeze(-1) + h

    # Consistency check
    consistent = (d_vecs == traj_d).all(dim=-1).all(dim=-1)

    # Mark NaN candidates as inconsistent
    nan_mask = torch.isnan(z_cands[:, 0])
    consistent = consistent & ~nan_mask

    return z_factors, traj, traj_d, consistent


def _full_iteration_kernel_optimised(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    d_vecs: Tensor,
    order: int,
    eye: Tensor,
    awd_table: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Optimised kernel: closed-form solve + optional AWD table."""
    B = d_vecs.shape[0]
    dim = A.shape[0]

    z_factors = eye.unsqueeze(0).expand(B, -1, -1).clone()
    h_factors = z_factors.clone()
    for k in range(order):
        if awd_table is not None:
            idx = d_vecs_to_indices(d_vecs[:, k, :])
            AWDs = awd_table[idx]
        else:
            AWDs = A.unsqueeze(0) + W.unsqueeze(0) * d_vecs[:, k, :].unsqueeze(1)
        z_factors = torch.bmm(AWDs, z_factors)
        if k > 0:
            h_factors = torch.bmm(AWDs, h_factors) + eye.unsqueeze(0)

    lhs = eye.unsqueeze(0) - z_factors
    rhs = (h_factors @ h).unsqueeze(-1)
    z_cands = batched_solve(lhs, rhs).squeeze(-1)

    traj = torch.empty(B, order, dim, dtype=A.dtype)
    traj_d = torch.empty(B, order, dim, dtype=A.dtype)
    z = z_cands
    for k in range(order):
        traj[:, k] = z
        d = (z > 0).to(A.dtype)
        traj_d[:, k] = d
        if awd_table is not None:
            idx = d_vecs_to_indices(d)
            AWD_sim = awd_table[idx]
        else:
            AWD_sim = A.unsqueeze(0) + W.unsqueeze(0) * d.unsqueeze(1)
        z = torch.bmm(AWD_sim, z.unsqueeze(-1)).squeeze(-1) + h

    consistent = (d_vecs == traj_d).all(dim=-1).all(dim=-1)
    nan_mask = torch.isnan(z_cands[:, 0])
    consistent = consistent & ~nan_mask

    return z_factors, traj, traj_d, consistent


# Compiled versions
try:
    _full_iteration_compiled = torch.compile(
        _full_iteration_kernel,
        fullgraph=False,
        dynamic=True,
    )
    _full_iteration_opt_compiled = torch.compile(
        _full_iteration_kernel_optimised,
        fullgraph=False,
        dynamic=True,
    )
    _HAS_COMPILE = True
except Exception:
    _full_iteration_compiled = _full_iteration_kernel
    _full_iteration_opt_compiled = _full_iteration_kernel_optimised
    _HAS_COMPILE = False


# ---------------------------------------------------------------------------
# PLRNN vectorised v2
# ---------------------------------------------------------------------------
def scy_fi_vectorised(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
    batch_size: int | None = None,
    compiled: bool = False,
    fast_solve: bool = False,
    precision: str = "float64",
    use_table: bool = False,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Vectorised search for k-cycles in a PLRNN.

    Processes ``batch_size`` candidates through the full pipeline
    every inner iteration without active-set tracking overhead.

    Parameters
    ----------
    A, W, h : Tensor
        PLRNN parameters.
    order : int
        Cycle order.
    found_lower_orders : list
        Previously found cycles.
    outer_loop_iterations, inner_loop_iterations : int or None
        Loop budgets.
    batch_size : int or None
        Parallel candidates. None = auto-tuned by dimension.
    compiled : bool
        Use torch.compile kernels.
    fast_solve : bool
        Use closed-form 2×2/3×3 solve instead of LAPACK.
    precision : str
        ``"float32"`` or ``"float64"``.
    use_table : bool
        Pre-compute AWD lookup table (dim ≤ 10 only).
    """
    dim = A.shape[0]
    dtype = torch.float32 if precision == "float32" else torch.float64
    A = A.to(dtype)
    W = W.to(dtype)
    h = h.to(dtype)
    B = batch_size if batch_size is not None else auto_batch_size(dim)

    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(
        order, outer_loop_iterations, inner_loop_iterations
    )

    eye = torch.eye(dim, dtype=dtype)

    # AWD table (opt 4)
    awd_table = None
    if use_table and dim <= 10:
        awd_table = build_awd_table(A, W)

    # Select kernel
    if fast_solve or awd_table is not None:
        kernel = (
            _full_iteration_opt_compiled
            if compiled
            else _full_iteration_kernel_optimised
        )
    else:
        kernel = _full_iteration_compiled if compiled else _full_iteration_kernel

    # Hash set for O(1) dedup
    _scale = 1000
    _use_fast_key = True
    found_keys: set[bytes | tuple[int, ...]] = set()
    for order_cycles in found_lower_orders:
        for t in order_cycles:
            for pt in t:
                found_keys.add(fast_key(pt.to(dtype), _scale))

    i = -1
    while i < outer_loop_iterations:
        i += 1

        # All B start with fresh random D-vecs
        d_vecs = torch.randint(0, 2, (B, order, dim), dtype=dtype)

        for _c in range(inner_loop_iterations):
            # === ONE kernel call does everything ===
            if awd_table is not None:
                z_factors, traj, traj_d, consistent = kernel(
                    A,
                    W,
                    h,
                    d_vecs,
                    order,
                    eye,
                    awd_table,
                )
            else:
                z_factors, traj, traj_d, consistent = kernel(
                    A,
                    W,
                    h,
                    d_vecs,
                    order,
                    eye,
                )

            # === Harvest consistent results (rare) ===
            c_indices = consistent.nonzero(as_tuple=False).squeeze(-1)
            for ci in c_indices:
                b = ci.item()
                z_pt = traj[b, 0]
                key0 = fast_key(z_pt, _scale)
                if key0 in found_keys:
                    continue

                trajectory = [traj[b, j].clone() for j in range(order)]
                if not _is_duplicate(trajectory, cycles_found):
                    # Eigvals — use torch, cast to float64 for stability
                    zf = torch.eye(dim, dtype=torch.float64)
                    d_f64 = d_vecs[b].to(torch.float64)
                    A_f64 = A.to(torch.float64)
                    W_f64 = W.to(torch.float64)
                    for k in range(order):
                        zf = (A_f64 + W_f64 * d_f64[k]) @ zf
                    eigvals = torch.linalg.eigvals(zf).numpy()

                    cycles_found.append(trajectory)
                    eigvals_found.append(eigvals)
                    for pt in trajectory:
                        found_keys.add(fast_key(pt, _scale))
                    i = 0

            # === Vectorised D-vec update ===
            # Check if all D-vecs already converged (early stop)
            all_converged = (d_vecs == traj_d).all()
            d_vecs[~consistent] = traj_d[~consistent]
            n_resample = consistent.sum().item()
            if n_resample > 0:
                d_vecs[consistent] = torch.randint(
                    0,
                    2,
                    (n_resample, order, dim),
                    dtype=dtype,
                )
            # Break inner loop if all D-vecs were already stable
            if all_converged:
                break

    return cycles_found, eigvals_found


# ---------------------------------------------------------------------------
# shPLRNN vectorised v2
# ---------------------------------------------------------------------------
def scy_fi_sh_vectorised(
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    h1: Tensor,
    h2: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
    batch_size: int = 64,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Vectorised search for k-cycles in a shPLRNN.

    Parameters
    ----------
    A : Tensor  Diagonal entries ``(M,)``.
    W1 : (M, H), W2 : (H, M), h1 : (M,), h2 : (H,)
    order, found_lower_orders, outer/inner, batch_size : as PLRNN.
    """
    latent_dim = A.shape[0]
    hidden_dim = h2.shape[0]
    dtype = A.dtype
    B = batch_size

    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(
        order, outer_loop_iterations, inner_loop_iterations
    )

    relu_pool = construct_relu_matrix_pool(A, W1, W2, h1, h2, latent_dim, hidden_dim)
    pool_d_vecs = torch.diagonal(relu_pool, dim1=1, dim2=2)
    pool_size = pool_d_vecs.shape[0]

    eye = torch.eye(latent_dim, dtype=dtype)
    A_diag = torch.diag(A)
    W2_B = W2.unsqueeze(0).expand(B, -1, -1)
    W1_B = W1.unsqueeze(0).expand(B, -1, -1)

    _scale = 1000
    found_keys: set[tuple[int, ...]] = set()
    for order_cycles in found_lower_orders:
        for t in order_cycles:
            for pt in t:
                found_keys.add(tuple(torch.round(pt * _scale).long().tolist()))

    i = -1
    while i < outer_loop_iterations:
        i += 1

        pool_indices = torch.randint(0, pool_size, (B, order))
        d_vecs = pool_d_vecs[pool_indices]

        for _c in range(inner_loop_iterations):
            # === Batched shPLRNN chain product ===
            z_factors = eye.unsqueeze(0).expand(B, -1, -1).clone()
            h1_factors = z_factors.clone()
            h2_factors = (W1.unsqueeze(0) * d_vecs[:, 0, :].unsqueeze(1)).clone()

            for k in range(order - 1):
                W1D = W1.unsqueeze(0) * d_vecs[:, k, :].unsqueeze(1)
                m = A_diag.unsqueeze(0) + torch.bmm(W1D, W2_B)
                z_factors = torch.bmm(m, z_factors)

                W1D_next = W1.unsqueeze(0) * d_vecs[:, k + 1, :].unsqueeze(1)
                m_next = A_diag.unsqueeze(0) + torch.bmm(W1D_next, W2_B)
                h1_factors = torch.bmm(m_next, h1_factors) + eye.unsqueeze(0)
                h2_factors = torch.bmm(m_next, h2_factors) + W1D_next

            W1D_last = W1.unsqueeze(0) * d_vecs[:, order - 1, :].unsqueeze(1)
            m_last = A_diag.unsqueeze(0) + torch.bmm(W1D_last, W2_B)
            z_factors = torch.bmm(m_last, z_factors)

            # === Batched solve ===
            lhs = eye.unsqueeze(0) - z_factors
            rhs_vec = torch.bmm(
                h1_factors, h1.unsqueeze(0).expand(B, -1).unsqueeze(-1)
            ) + torch.bmm(h2_factors, h2.unsqueeze(0).expand(B, -1).unsqueeze(-1))
            try:
                z_cands = torch.linalg.solve(lhs, rhs_vec).squeeze(-1)
            except torch.linalg.LinAlgError:
                z_cands = torch.full((B, latent_dim), float("nan"), dtype=dtype)
                for b in range(B):
                    try:
                        z_cands[b] = torch.linalg.solve(lhs[b], rhs_vec[b]).squeeze(-1)
                    except torch.linalg.LinAlgError:
                        pass

            # === Batched simulation ===
            traj = torch.empty(B, order, latent_dim, dtype=dtype)
            traj_d = torch.empty(B, order, hidden_dim, dtype=dtype)
            z = z_cands
            for k in range(order):
                traj[:, k] = z
                hidden_act = torch.bmm(W2_B, z.unsqueeze(-1)).squeeze(-1) + h2
                d_k = (hidden_act > 0).to(dtype)
                traj_d[:, k] = d_k
                z = (
                    A.unsqueeze(0) * z
                    + torch.bmm(
                        W1_B, torch.clamp(hidden_act, min=0.0).unsqueeze(-1)
                    ).squeeze(-1)
                    + h1
                )

            # === Consistency + harvest ===
            consistent = (d_vecs == traj_d).all(dim=-1).all(dim=-1)
            nan_mask = torch.isnan(z_cands[:, 0])
            consistent = consistent & ~nan_mask

            c_indices = consistent.nonzero(as_tuple=False).squeeze(-1)
            for ci in c_indices:
                b = ci.item()
                z_pt = traj[b, 0]
                key0 = tuple(torch.round(z_pt * _scale).long().tolist())
                if key0 in found_keys:
                    continue
                trajectory = [traj[b, j].clone() for j in range(order)]
                if not _is_duplicate(trajectory, cycles_found):
                    zf = eye.clone()
                    for k in range(order):
                        W1Dk = W1 * d_vecs[b, k]
                        zf = (A_diag + W1Dk @ W2) @ zf
                    eigvals = np.linalg.eigvals(zf.numpy())
                    cycles_found.append(trajectory)
                    eigvals_found.append(eigvals)
                    for pt in trajectory:
                        found_keys.add(tuple(torch.round(pt * _scale).long().tolist()))
                    i = 0

            # Update D-vecs
            d_vecs[~consistent] = traj_d[~consistent]
            n_resample = consistent.sum().item()
            if n_resample > 0:
                pi = torch.randint(0, pool_size, (n_resample, order))
                d_vecs[consistent] = pool_d_vecs[pi]

    return cycles_found, eigvals_found


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------
def find_cycles_vectorised(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int,
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
    batch_size: int | None = None,
    compiled: bool = False,
    fast_solve: bool = False,
    precision: str = "float64",
    use_table: bool = False,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find all cycles up to ``max_order`` in a PLRNN (vectorised).

    Parameters
    ----------
    A, W, h : Tensor
        PLRNN parameters.
    max_order, outer_loop_iterations, inner_loop_iterations : int
        Search budgets.
    batch_size : int or None
        Parallel candidates. None = auto by dim.
    compiled : bool
        Use torch.compile kernels.
    fast_solve : bool
        Closed-form 2×2/3×3 solve.
    precision : str
        ``"float32"`` or ``"float64"``.
    use_table : bool
        Pre-compute AWD lookup table (dim ≤ 10).
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi_vectorised(
            A,
            W,
            h,
            order,
            all_cycles,
            outer_loop_iterations=outer_loop_iterations,
            inner_loop_iterations=inner_loop_iterations,
            batch_size=batch_size,
            compiled=compiled,
            fast_solve=fast_solve,
            precision=precision,
            use_table=use_table,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals


def find_cycles_sh_vectorised(
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    h1: Tensor,
    h2: Tensor,
    max_order: int,
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
    batch_size: int = 64,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find all cycles up to ``max_order`` in a shPLRNN (vectorised).

    Parameters
    ----------
    A : (M,), W1 : (M,H), W2 : (H,M), h1 : (M,), h2 : (H,)
    max_order, outer/inner, batch_size : search budgets.
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi_sh_vectorised(
            A,
            W1,
            W2,
            h1,
            h2,
            order,
            all_cycles,
            outer_loop_iterations=outer_loop_iterations,
            inner_loop_iterations=inner_loop_iterations,
            batch_size=batch_size,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals
