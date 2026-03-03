"""Woodbury SCYFI — structured solve for fixed points (k=1).

Exploits the structure (I - A - W*D) where A is diagonal:
(I-A) is trivially invertible, and W*D is a rank-r perturbation
(r = number of active bits in D). Uses the Woodbury identity
for an O(dim*r) solve instead of O(dim^3).
"""
# ruff: noqa: E501

from __future__ import annotations

import torch
from numpy import ndarray
from torch import Tensor

from dynamic.analysis.scyfi import _is_duplicate
from dynamic.analysis.scyfi_solve import fast_key


def _woodbury_solve_single(
    inv_I_minus_A: Tensor,
    W: Tensor,
    d: Tensor,
    h: Tensor,
) -> Tensor:
    """Solve (I - A - W*D) z = h using Woodbury identity.

    (I-A-WD) = (I-A)(I - (I-A)^{-1} WD)

    By Woodbury:
    (I-A-WD)^{-1} = (I-A)^{-1} + (I-A)^{-1} U (I - V (I-A)^{-1} U)^{-1} V (I-A)^{-1}

    where U = W[:, active], V = D[active, :] (selecting active rows).

    Parameters
    ----------
    inv_I_minus_A : (dim,) — diagonal of (I-A)^{-1}
    W : (dim, dim)
    d : (dim,) — binary D-vector
    h : (dim,) — bias

    Returns
    -------
    Tensor : (dim,) — solution z
    """
    dim = d.shape[0]
    active = (d > 0.5).nonzero(as_tuple=False).squeeze(-1)
    r = active.shape[0]

    # (I-A)^{-1} h
    inv_h = inv_I_minus_A * h  # (dim,)

    if r == 0:
        return inv_h

    # U = W[:, active]  (dim × r)
    U = W[:, active]

    # V = I[active, :]  (r × dim) — just row selection
    # (I-A)^{-1} U  (dim × r)
    inv_U = inv_I_minus_A.unsqueeze(-1) * U  # (dim, r)

    # Core: I_r - V @ inv_U = I_r - inv_U[active, :]  (r × r)
    core = torch.eye(r, dtype=d.dtype) - inv_U[active, :]

    # Solve core system
    try:
        # core @ x = V @ inv_h = inv_h[active]
        rhs_core = inv_h[active]  # (r,)
        x = torch.linalg.solve(core, rhs_core.unsqueeze(-1)).squeeze(-1)
    except torch.linalg.LinAlgError:
        return torch.full((dim,), float("nan"), dtype=d.dtype)

    # Full solution: inv_h + inv_U @ x
    return inv_h + inv_U @ x


def _woodbury_solve_batched(
    inv_I_minus_A: Tensor,
    W: Tensor,
    d_vecs: Tensor,
    h: Tensor,
) -> Tensor:
    """Batched Woodbury solve for B D-vectors.

    Parameters
    ----------
    inv_I_minus_A : (dim,)
    W : (dim, dim)
    d_vecs : (B, dim)
    h : (dim,)

    Returns
    -------
    Tensor : (B, dim)
    """
    B = d_vecs.shape[0]
    dim = h.shape[0]
    result = torch.empty(B, dim, dtype=h.dtype)

    for i in range(B):
        result[i] = _woodbury_solve_single(inv_I_minus_A, W, d_vecs[i], h)

    return result


def scy_fi_woodbury(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
    B: int = 64,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Find fixed points (k=1) using Woodbury structured solve.

    Parameters
    ----------
    A, W, h : Tensor
        PLRNN parameters.
    found_lower_orders : list
        Previously found cycles.
    B : int
        Number of parallel D-vector candidates.
    """
    dim = A.shape[0]
    dtype = A.dtype

    if outer_loop_iterations is None:
        outer_loop_iterations = 10
    if inner_loop_iterations is None:
        inner_loop_iterations = 20

    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    # Pre-compute (I - A)^{-1} diagonal
    diag_A = A.diag() if A.dim() == 2 else A
    inv_I_minus_A = 1.0 / (1.0 - diag_A)

    _scale = 1000
    found_keys: set[bytes] = set()
    for order_cycles in found_lower_orders:
        for t in order_cycles:
            for pt in t:
                found_keys.add(fast_key(pt, _scale))

    i = -1
    while i < outer_loop_iterations:
        i += 1
        d_vecs = torch.randint(0, 2, (B, dim), dtype=dtype)

        for _c in range(inner_loop_iterations):
            # Woodbury solve for all B
            z_cands = _woodbury_solve_batched(
                inv_I_minus_A,
                W,
                d_vecs,
                h,
            )

            # Simulate: check if D(z) == proposed D
            d_observed = (z_cands > 0).to(dtype)
            consistent = (d_vecs == d_observed).all(dim=-1)
            nan_mask = torch.isnan(z_cands[:, 0])
            consistent = consistent & ~nan_mask

            # Harvest
            c_indices = consistent.nonzero(as_tuple=False).squeeze(-1)
            for ci in c_indices:
                b = ci.item()
                z_pt = z_cands[b]
                key0 = fast_key(z_pt, _scale)
                if key0 in found_keys:
                    d_vecs[b] = torch.randint(0, 2, (dim,), dtype=dtype)
                    continue

                trajectory = [z_pt.clone()]
                if not _is_duplicate(trajectory, cycles_found):
                    zf = A + W * d_vecs[b]
                    eigvals = torch.linalg.eigvals(zf).numpy()
                    cycles_found.append(trajectory)
                    eigvals_found.append(eigvals)
                    found_keys.add(key0)
                    i = 0

                d_vecs[b] = torch.randint(0, 2, (dim,), dtype=dtype)

            # Refine inconsistent
            d_vecs[~consistent] = d_observed[~consistent]

    return cycles_found, eigvals_found


def find_cycles_woodbury(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int = 1,
    *,
    B: int = 64,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find fixed points using Woodbury. Only supports k=1.

    For k > 1, returns empty lists for those orders.
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    # Order 1: Woodbury
    cycles, eigvals = scy_fi_woodbury(
        A,
        W,
        h,
        all_cycles,
        B=B,
        outer_loop_iterations=outer_loop_iterations,
        inner_loop_iterations=inner_loop_iterations,
    )
    all_cycles.append(cycles)
    all_eigvals.append(eigvals)

    # Orders > 1: empty (Woodbury only handles k=1)
    for _ in range(2, max_order + 1):
        all_cycles.append([])
        all_eigvals.append([])

    return all_cycles, all_eigvals
