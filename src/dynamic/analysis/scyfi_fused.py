"""Fully fused SCYFI algorithm — zero helper function calls.

All helper computations (chain product, candidate solve, simulation,
D-extraction, consistency check) are inlined directly into the search
loop body to eliminate Python function-call overhead.

The reference implementations in ``scyfi.py`` and the intermediate
optimised versions in ``scyfi_fast.py`` remain untouched.
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


# ---------------------------------------------------------------------------
# PLRNN fused
# ---------------------------------------------------------------------------
def scy_fi_fused(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Fully fused heuristic search for k-cycles in a PLRNN.

    Every helper computation is inlined — no function calls in the
    inner loop except ``torch.linalg.inv`` and ``torch.equal``.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)`` — diagonal matrix.
    W : Tensor
        Shape ``(M, M)`` — weight matrix.
    h : Tensor
        Shape ``(M,)`` — bias vector.
    order : int
        Cycle order to search for.
    found_lower_orders : list
        Previously found cycles at orders 1..order-1.
    outer_loop_iterations : int or None
        Number of random restarts.
    inner_loop_iterations : int or None
        Number of inner refinement steps.

    Returns
    -------
    tuple[list, list]
        ``(cycles_found, eigvals_found)``.
    """
    dim = A.shape[0]
    dtype = A.dtype
    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(
        order, outer_loop_iterations, inner_loop_iterations
    )

    # Pre-allocate — reused every iteration
    eye = torch.eye(dim, dtype=dtype)
    z_factor = torch.empty(dim, dim, dtype=dtype)
    h_factor = torch.empty(dim, dim, dtype=dtype)
    traj = torch.empty(order, dim, dtype=dtype)
    traj_d = torch.empty(order, dim, dtype=dtype)

    # Hash set for O(1) dedup
    found_keys: set[tuple[int, ...]] = set()
    _scale = 1000
    for order_cycles in found_lower_orders:
        for t in order_cycles:
            for pt in t:
                found_keys.add(
                    tuple(torch.round(pt * _scale).long().tolist())
                )

    i = -1
    while i < outer_loop_iterations:
        i += 1
        # Inline random D-vec generation
        d_vecs = torch.randint(0, 2, (order, dim), dtype=dtype)
        c = 0

        while c < inner_loop_iterations:
            c += 1

            # === Inline chain product: z_factor, h_factor ===
            z_factor.copy_(eye)
            h_factor.copy_(eye)
            for k in range(order):
                AWD = A + W * d_vecs[k]
                z_factor = AWD @ z_factor
                if k > 0:
                    h_factor = AWD @ h_factor + eye

            # === Inline candidate solve ===
            try:
                inv_mat = torch.linalg.inv(eye - z_factor)
            except torch.linalg.LinAlgError:
                d_vecs = torch.randint(0, 2, (order, dim), dtype=dtype)
                continue
            z_cand = inv_mat @ (h_factor @ h)

            # === Inline simulate + extract D-vecs ===
            z = z_cand
            for k in range(order):
                traj[k] = z
                d = (z > 0).to(dtype)
                traj_d[k] = d
                # A @ z + (W * d) @ z + h  ≡  A @ z + W @ clamp(z, 0) + h
                z = A @ z + (W * d) @ z + h

            # === Inline consistency check ===
            consistent = torch.equal(d_vecs, traj_d)

            if consistent:
                # O(1) lower-order check
                key0 = tuple(
                    torch.round(traj[0] * _scale).long().tolist()
                )
                if key0 in found_keys:
                    d_vecs = torch.randint(
                        0, 2, (order, dim), dtype=dtype
                    )
                    continue

                # Duplicate check at this order
                trajectory = [traj[j].clone() for j in range(order)]
                if not _is_duplicate(trajectory, cycles_found):
                    # Eigvals — recompute z_factor (only on success)
                    zf = eye.clone()
                    for k in range(order):
                        zf = (A + W * d_vecs[k]) @ zf
                    eigvals = np.linalg.eigvals(zf.numpy())

                    cycles_found.append(trajectory)
                    eigvals_found.append(eigvals)

                    for pt in trajectory:
                        found_keys.add(
                            tuple(
                                torch.round(pt * _scale).long().tolist()
                            )
                        )
                    i = 0
                    c = 0

                # Resample
                d_vecs = torch.randint(0, 2, (order, dim), dtype=dtype)
            else:
                # Use observed D-vecs for next refinement iteration
                d_vecs = traj_d.clone()

    return cycles_found, eigvals_found


# ---------------------------------------------------------------------------
# shPLRNN fused
# ---------------------------------------------------------------------------
def scy_fi_sh_fused(
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
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Fully fused heuristic search for k-cycles in a shPLRNN.

    Parameters
    ----------
    A : Tensor
        Diagonal entries ``(M,)``.
    W1 : Tensor
        Shape ``(M, H)``.
    W2 : Tensor
        Shape ``(H, M)``.
    h1 : Tensor
        Shape ``(M,)``.
    h2 : Tensor
        Shape ``(H,)``.
    order : int
        Cycle order.
    found_lower_orders : list
        Previously found cycles.
    outer_loop_iterations : int or None
        Number of random restarts.
    inner_loop_iterations : int or None
        Number of inner refinement steps.

    Returns
    -------
    tuple[list, list]
        ``(cycles_found, eigvals_found)``.
    """
    latent_dim = A.shape[0]
    hidden_dim = h2.shape[0]
    dtype = A.dtype

    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(
        order, outer_loop_iterations, inner_loop_iterations
    )

    # Build pool of reachable D patterns
    relu_pool = construct_relu_matrix_pool(
        A, W1, W2, h1, h2, latent_dim, hidden_dim
    )
    pool_d_vecs = torch.diagonal(relu_pool, dim1=1, dim2=2)  # (N, H)
    pool_size = pool_d_vecs.shape[0]

    # Pre-allocate
    eye = torch.eye(latent_dim, dtype=dtype)
    A_diag = torch.diag(A)
    traj = torch.empty(order, latent_dim, dtype=dtype)
    traj_d = torch.empty(order, hidden_dim, dtype=dtype)

    # Hash set
    found_keys: set[tuple[int, ...]] = set()
    _scale = 1000
    for order_cycles in found_lower_orders:
        for t in order_cycles:
            for pt in t:
                found_keys.add(
                    tuple(torch.round(pt * _scale).long().tolist())
                )

    i = -1
    while i < outer_loop_iterations:
        i += 1
        indices = torch.randint(0, pool_size, (order,))
        d_vecs = pool_d_vecs[indices]  # (order, H)
        c = 0

        while c < inner_loop_iterations:
            c += 1

            # === Inline shPLRNN chain product ===
            z_factor = eye.clone()
            h1_factor = eye.clone()
            h2_factor = (W1 * d_vecs[0]).clone()  # (M, H)

            for k in range(order - 1):
                W1D = W1 * d_vecs[k]         # (M, H)
                m = A_diag + W1D @ W2         # (M, M)
                z_factor = m @ z_factor

                W1D_next = W1 * d_vecs[k + 1]
                m_next = A_diag + W1D_next @ W2
                h1_factor = m_next @ h1_factor + eye
                h2_factor = m_next @ h2_factor + W1D_next

            # Final z_factor step
            W1D_last = W1 * d_vecs[order - 1]
            m_last = A_diag + W1D_last @ W2
            z_factor = m_last @ z_factor

            # === Inline candidate solve ===
            try:
                inv_mat = torch.linalg.inv(eye - z_factor)
            except torch.linalg.LinAlgError:
                indices = torch.randint(0, pool_size, (order,))
                d_vecs = pool_d_vecs[indices]
                continue
            z_cand = inv_mat @ (h1_factor @ h1 + h2_factor @ h2)

            # === Inline simulate + extract D-vecs ===
            z = z_cand
            for k in range(order):
                traj[k] = z
                hidden_act = W2 @ z + h2
                traj_d[k] = (hidden_act > 0).to(dtype)
                z = A * z + W1 @ torch.clamp(hidden_act, min=0.0) + h1

            # === Inline consistency check ===
            consistent = torch.equal(d_vecs, traj_d)

            if consistent:
                key0 = tuple(
                    torch.round(traj[0] * _scale).long().tolist()
                )
                if key0 in found_keys:
                    indices = torch.randint(0, pool_size, (order,))
                    d_vecs = pool_d_vecs[indices]
                    continue

                trajectory = [traj[j].clone() for j in range(order)]
                if not _is_duplicate(trajectory, cycles_found):
                    # Eigvals on success only
                    zf = eye.clone()
                    for k in range(order):
                        W1Dk = W1 * d_vecs[k]
                        zf = (A_diag + W1Dk @ W2) @ zf
                    eigvals = np.linalg.eigvals(zf.numpy())

                    cycles_found.append(trajectory)
                    eigvals_found.append(eigvals)

                    for pt in trajectory:
                        found_keys.add(
                            tuple(
                                torch.round(pt * _scale).long().tolist()
                            )
                        )
                    i = 0
                    c = 0

                indices = torch.randint(0, pool_size, (order,))
                d_vecs = pool_d_vecs[indices]
            else:
                d_vecs = traj_d.clone()

    return cycles_found, eigvals_found


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------
def find_cycles_fused(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int,
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find all cycles up to a given order in a PLRNN (fully fused).

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    h : Tensor
        Shape ``(M,)``.
    max_order : int
        Maximum cycle order.
    outer_loop_iterations : int or None
        Random restarts per order.
    inner_loop_iterations : int or None
        Inner refinement steps per order.

    Returns
    -------
    tuple[list, list]
        ``(all_cycles, all_eigvals)``.
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi_fused(
            A, W, h, order, all_cycles,
            outer_loop_iterations=outer_loop_iterations,
            inner_loop_iterations=inner_loop_iterations,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals


def find_cycles_sh_fused(
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    h1: Tensor,
    h2: Tensor,
    max_order: int,
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find all cycles up to a given order in a shPLRNN (fully fused).

    Parameters
    ----------
    A : Tensor
        Diagonal entries ``(M,)``.
    W1 : Tensor
        Shape ``(M, H)``.
    W2 : Tensor
        Shape ``(H, M)``.
    h1 : Tensor
        Shape ``(M,)``.
    h2 : Tensor
        Shape ``(H,)``.
    max_order : int
        Maximum cycle order.
    outer_loop_iterations : int or None
        Random restarts.
    inner_loop_iterations : int or None
        Inner refinement steps.

    Returns
    -------
    tuple[list, list]
        ``(all_cycles, all_eigvals)``.
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi_sh_fused(
            A, W1, W2, h1, h2, order, all_cycles,
            outer_loop_iterations=outer_loop_iterations,
            inner_loop_iterations=inner_loop_iterations,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals
