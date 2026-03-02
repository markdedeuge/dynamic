"""Exhaustive search for cycles in PLRNNs.

Brute-force enumeration over all subregion sequences. Only practical
for small dimensions. Used as ground-truth verification against the
heuristic SCYFI algorithm.

Port of ``reference/SCYFI/src/utilities/exhaustive_search.jl``.
"""

from __future__ import annotations

import itertools

import torch
from numpy import ndarray
from torch import Tensor

from dynamic.analysis.scyfi_helpers import (
    construct_relu_matrix,
    get_cycle_point_candidate,
    get_eigvals,
    get_latent_time_series,
)


def _is_duplicate_exhaustive(
    z: Tensor,
    existing: list[list[Tensor]],
    digits: int = 2,
) -> bool:
    """Check if z (rounded) already exists in any trajectory."""
    scale = 10**digits
    z_q = torch.round(z * scale).long()
    for traj in existing:
        for point in traj:
            p_q = torch.round(point * scale).long()
            if torch.equal(z_q, p_q):
                return True
    return False


def _is_in_lower_orders_exhaustive(
    z: Tensor,
    found_lower_orders: list[list[list[Tensor]]],
    digits: int = 2,
) -> bool:
    """Check if z was already found in a lower-order cycle."""
    scale = 10**digits
    z_q = torch.round(z * scale).long()
    for order_cycles in found_lower_orders:
        for traj in order_cycles:
            for point in traj:
                p_q = torch.round(point * scale).long()
                if torch.equal(z_q, p_q):
                    return True
    return False


def exhaustive_search(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
) -> tuple[list[list[Tensor]], list[ndarray], list[int]]:
    """Brute-force search over all subregion sequences.

    Enumerates all ``(2^dim)^order`` possible D-matrix sequences,
    solves for the candidate cycle point, and checks self-consistency.

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

    Returns
    -------
    tuple[list, list, list]
        ``(cycles_found, eigvals, idx_found)``.
    """
    dim = A.shape[0]
    n = 2**dim
    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []
    idx_found: list[int] = []

    # Enumerate all combinations of subregion indices
    for idx, item in enumerate(itertools.product(range(n), repeat=order)):
        # Build D_list from quadrant indices
        D_list = torch.zeros(order, dim, dim)
        for i in range(order):
            D_list[i] = construct_relu_matrix(item[i], dim)

        z_candidate = get_cycle_point_candidate(A, W, D_list, h, order)
        if z_candidate is None:
            continue

        trajectory = get_latent_time_series(
            order, A, W, h, dim, z_0=z_candidate
        )

        # Build D matrices from trajectory
        traj_relu = torch.zeros(order, dim, dim)
        for j in range(order):
            traj_relu[j] = torch.diag((trajectory[j] > 0).float())

        # Check self-consistency
        is_consistent = True
        for j in range(order):
            diff = torch.sum(
                torch.abs(traj_relu[j] - D_list[j])
            ).item()
            if diff != 0:
                is_consistent = False
                break

            if found_lower_orders and _is_in_lower_orders_exhaustive(
                trajectory[0], found_lower_orders
            ):
                is_consistent = False
                break

        if is_consistent:
            if not _is_duplicate_exhaustive(trajectory[0], cycles_found):
                e = get_eigvals(A, W, D_list, order)
                cycles_found.append(trajectory)
                eigvals_found.append(e)
                idx_found.append(idx)

    return cycles_found, eigvals_found, idx_found


def main_exhaustive(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Run exhaustive search for all orders up to max_order.

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

    Returns
    -------
    tuple[list, list]
        ``(all_cycles, all_eigvals)`` indexed by order.
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals, _ = exhaustive_search(
            A, W, h, order, all_cycles
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals
