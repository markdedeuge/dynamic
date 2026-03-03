"""Optimised SCYFI algorithm — find_cycles_fast.

Performance-focused reimplementation of ``scyfi.py``.
Uses batched candidates, fused chain products, boolean D-vectors,
and hash-based deduplication.  The reference implementations in
``scyfi.py`` and ``scyfi_helpers.py`` remain untouched.
"""

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
from dynamic.analysis.scyfi_helpers_fast import (
    _chain_product,
    _chain_product_sh,
    batch_candidates,
    make_key,
    random_d_vecs,
    random_d_vecs_batch,
    simulate_and_extract,
    simulate_and_extract_sh,
)


# ---------------------------------------------------------------------------
# Lean inner-loop helpers (avoid computing what you don't need)
# ---------------------------------------------------------------------------
def _get_candidate_only(
    A: Tensor, W: Tensor, d_vecs: Tensor, h: Tensor, order: int, eye: Tensor,
) -> Tensor | None:
    """Chain product → candidate only (no eigvals)."""
    z_factor, h_factor = _chain_product(A, W, d_vecs, order, eye)
    try:
        inv_mat = torch.linalg.inv(eye - z_factor)
        return inv_mat @ (h_factor @ h)
    except torch.linalg.LinAlgError:
        return None


def _get_eigvals_only(
    A: Tensor, W: Tensor, d_vecs: Tensor, order: int, eye: Tensor,
) -> ndarray:
    """Chain product → eigvals only (no candidate)."""
    z_factor, _ = _chain_product(A, W, d_vecs, order, eye)
    return np.linalg.eigvals(z_factor.numpy())


def _get_candidate_only_sh(
    A: Tensor, W1: Tensor, W2: Tensor,
    h1: Tensor, h2: Tensor,
    d_vecs: Tensor, order: int, eye: Tensor,
) -> Tensor | None:
    """shPLRNN chain product → candidate only (no eigvals)."""
    z_factor, h1_factor, h2_factor = _chain_product_sh(
        A, W1, W2, d_vecs, order, eye,
    )
    try:
        inv_mat = torch.linalg.inv(eye - z_factor)
        return inv_mat @ (h1_factor @ h1 + h2_factor @ h2)
    except torch.linalg.LinAlgError:
        return None


def _get_eigvals_only_sh(
    A: Tensor, W1: Tensor, W2: Tensor,
    d_vecs: Tensor, order: int, eye: Tensor,
) -> ndarray:
    """shPLRNN chain product → eigvals only (no candidate)."""
    z_factor, _, _ = _chain_product_sh(
        A, W1, W2, d_vecs, order, eye,
    )
    return np.linalg.eigvals(z_factor.numpy())


# ---------------------------------------------------------------------------
# PLRNN fast implementation
# ---------------------------------------------------------------------------
def scy_fi_fast(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Optimised heuristic search for k-cycles in a PLRNN.

    Same API as ``scy_fi()`` but uses:
    - Boolean D-vectors instead of full diagonal matrices
    - Fused candidate + eigenvalue computation
    - Fused simulate + D-extraction
    - Hash-based duplicate detection
    - Pre-cached identity matrix

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    h : Tensor
        Shape ``(M,)``.
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
    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(
        order, outer_loop_iterations, inner_loop_iterations
    )

    # Pre-allocate identity
    eye = torch.eye(dim, dtype=A.dtype)

    # Build set of known keys for O(1) dedup
    found_keys: set[tuple[int, ...]] = set()
    for order_cycles in found_lower_orders:
        for traj in order_cycles:
            for point in traj:
                found_keys.add(make_key(point))

    i = -1
    while i < outer_loop_iterations:
        i += 1
        d_vecs = random_d_vecs(dim, order, dtype=A.dtype)
        c = 0

        while c < inner_loop_iterations:
            c += 1

            # Candidate only — skip eigvals (saves ~40% per call)
            z_candidate = _get_candidate_only(
                A, W, d_vecs, h, order, eye,
            )

            if z_candidate is not None:
                # Fused: simulate trajectory AND extract D-vectors
                traj, traj_d_vecs = simulate_and_extract(
                    z_candidate, A, W, h, order,
                )

                # Self-consistency check (vectorised)
                consistent = torch.equal(d_vecs, traj_d_vecs)

                if consistent:
                    # Check not in lower orders (O(1) hash lookup)
                    traj_key = make_key(traj[0])
                    if traj_key in found_keys:
                        # Already known — resample
                        d_vecs = random_d_vecs(dim, order, dtype=A.dtype)
                        continue

                    # Check not duplicate at this order
                    trajectory = [traj[j] for j in range(order)]
                    if not _is_duplicate(trajectory, cycles_found):
                        # Only compute eigvals on success
                        eigvals = _get_eigvals_only(
                            A, W, d_vecs, order, eye,
                        )
                        cycles_found.append(trajectory)
                        eigvals_found.append(eigvals)

                        # Register all trajectory points
                        for pt in trajectory:
                            found_keys.add(make_key(pt))

                        i = 0
                        c = 0

                # Update D-vectors for next iteration
                if consistent:
                    d_vecs = random_d_vecs(dim, order, dtype=A.dtype)
                else:
                    d_vecs = traj_d_vecs
            else:
                d_vecs = random_d_vecs(dim, order, dtype=A.dtype)

    return cycles_found, eigvals_found


# ---------------------------------------------------------------------------
# PLRNN batched fast implementation
# ---------------------------------------------------------------------------
def scy_fi_batched(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
    batch_size: int = 32,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Batched heuristic search for k-cycles in a PLRNN.

    Processes ``batch_size`` random D-vector sequences in parallel
    using ``torch.bmm``.  Falls back to sequential refinement on
    promising candidates.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    h : Tensor
        Shape ``(M,)``.
    order : int
        Cycle order.
    found_lower_orders : list
        Previously found cycles.
    outer_loop_iterations : int or None
        Number of batched random restarts.
    inner_loop_iterations : int or None
        Max inner refinement steps per candidate.
    batch_size : int
        Number of parallel candidates per outer iteration.

    Returns
    -------
    tuple[list, list]
        ``(cycles_found, eigvals_found)``.
    """
    dim = A.shape[0]
    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(
        order, outer_loop_iterations, inner_loop_iterations
    )

    eye = torch.eye(dim, dtype=A.dtype)

    # Hash set for O(1) dedup
    found_keys: set[tuple[int, ...]] = set()
    for order_cycles in found_lower_orders:
        for traj in order_cycles:
            for point in traj:
                found_keys.add(make_key(point))

    i = -1
    while i < outer_loop_iterations:
        i += 1

        # Phase 1: Batched candidate generation
        d_batch = random_d_vecs_batch(
            batch_size, dim, order, dtype=A.dtype,
        )
        candidates, z_factors, valid = batch_candidates(
            A, W, h, d_batch, order, eye,
        )

        # Phase 2: Sequential refinement of valid candidates
        for b in range(batch_size):
            if not valid[b]:
                continue

            z_cand = candidates[b]
            d_vecs = d_batch[b]  # (order, dim)

            for _ in range(inner_loop_iterations):
                traj, traj_d_vecs = simulate_and_extract(
                    z_cand, A, W, h, order,
                )
                consistent = torch.equal(d_vecs, traj_d_vecs)

                if consistent:
                    traj_key = make_key(traj[0])
                    if traj_key in found_keys:
                        break

                    trajectory = [traj[j] for j in range(order)]
                    if not _is_duplicate(trajectory, cycles_found):
                        eigvals = _get_eigvals_only(
                            A, W, d_vecs, order, eye,
                        )
                        cycles_found.append(trajectory)
                        eigvals_found.append(eigvals)

                        for pt in trajectory:
                            found_keys.add(make_key(pt))

                        i = 0
                    break  # Consistent — no more refinement needed

                # Not consistent — refine with new D-vecs
                d_vecs = traj_d_vecs
                z_cand_new = _get_candidate_only(
                    A, W, d_vecs, h, order, eye,
                )
                if z_cand_new is None:
                    break
                z_cand = z_cand_new

    return cycles_found, eigvals_found


# ---------------------------------------------------------------------------
# shPLRNN fast implementation
# ---------------------------------------------------------------------------
def scy_fi_sh_fast(
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
    """Optimised heuristic search for k-cycles in a shPLRNN.

    Same API as ``scy_fi_sh()`` but with fused operations and
    hash-based dedup.

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

    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(
        order, outer_loop_iterations, inner_loop_iterations
    )

    # Build pool of allowed D patterns (reuse reference impl — already fast)
    relu_pool = construct_relu_matrix_pool(
        A, W1, W2, h1, h2, latent_dim, hidden_dim
    )
    # Extract diagonal vectors from pool: (N, H, H) → (N, H)
    pool_d_vecs = torch.diagonal(relu_pool, dim1=1, dim2=2)  # (N, H)

    eye = torch.eye(latent_dim, dtype=A.dtype)

    found_keys: set[tuple[int, ...]] = set()
    for order_cycles in found_lower_orders:
        for traj in order_cycles:
            for point in traj:
                found_keys.add(make_key(point))

    i = -1
    while i < outer_loop_iterations:
        i += 1

        # Random D-vectors from pool
        indices = torch.randint(0, pool_d_vecs.shape[0], (order,))
        d_vecs = pool_d_vecs[indices]  # (order, H)

        c = 0
        while c < inner_loop_iterations:
            c += 1

            z_candidate = _get_candidate_only_sh(
                A, W1, W2, h1, h2, d_vecs, order, eye,
            )

            if z_candidate is not None:
                traj, traj_d_vecs = simulate_and_extract_sh(
                    z_candidate, A, W1, W2, h1, h2, order,
                )
                consistent = torch.equal(d_vecs, traj_d_vecs)

                if consistent:
                    traj_key = make_key(traj[0])
                    if traj_key in found_keys:
                        indices = torch.randint(
                            0, pool_d_vecs.shape[0], (order,)
                        )
                        d_vecs = pool_d_vecs[indices]
                        continue

                    trajectory = [traj[j] for j in range(order)]
                    if not _is_duplicate(trajectory, cycles_found):
                        eigvals = _get_eigvals_only_sh(
                            A, W1, W2, d_vecs, order, eye,
                        )
                        cycles_found.append(trajectory)
                        eigvals_found.append(eigvals)

                        for pt in trajectory:
                            found_keys.add(make_key(pt))

                        i = 0
                        c = 0

                if consistent:
                    indices = torch.randint(
                        0, pool_d_vecs.shape[0], (order,)
                    )
                    d_vecs = pool_d_vecs[indices]
                else:
                    d_vecs = traj_d_vecs
            else:
                indices = torch.randint(
                    0, pool_d_vecs.shape[0], (order,)
                )
                d_vecs = pool_d_vecs[indices]

    return cycles_found, eigvals_found


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------
def find_cycles_fast(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int,
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
    batched: bool = False,
    batch_size: int = 32,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find all cycles up to a given order in a PLRNN (optimised).

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
    batched : bool
        If True, use batched bmm candidate generation.
    batch_size : int
        Batch size for batched mode.

    Returns
    -------
    tuple[list, list]
        ``(all_cycles, all_eigvals)``.
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    fn = scy_fi_batched if batched else scy_fi_fast

    for order in range(1, max_order + 1):
        kwargs = {
            "outer_loop_iterations": outer_loop_iterations,
            "inner_loop_iterations": inner_loop_iterations,
        }
        if batched:
            kwargs["batch_size"] = batch_size

        cycles, eigvals = fn(
            A, W, h, order, all_cycles, **kwargs,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals


def find_cycles_sh_fast(
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
    """Find all cycles up to a given order in a shPLRNN (optimised).

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
        cycles, eigvals = scy_fi_sh_fast(
            A, W1, W2, h1, h2, order, all_cycles,
            outer_loop_iterations=outer_loop_iterations,
            inner_loop_iterations=inner_loop_iterations,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals
