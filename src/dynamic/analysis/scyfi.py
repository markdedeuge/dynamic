"""SCYFI — Searcher for CYcles and FIxed points.

Port of ``reference/SCYFI/src/scyfi_algo/`` from Julia to Python/PyTorch.
Finds all fixed points and k-cycles in ReLU-based PLRNNs by exploiting
their piecewise-linear structure.

Reference:
    Eisenmann L., Monfared M., Göring N., Durstewitz D.
    "Bifurcations and loss jumps in RNN training"
    NeurIPS 2023.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from dynamic.analysis.scyfi_helpers import (
    construct_relu_matrix_list,
    construct_relu_matrix_list_from_pool,
    construct_relu_matrix_pool,
    get_cycle_point_candidate,
    get_cycle_point_candidate_sh,
    get_eigvals,
    get_eigvals_sh,
    get_latent_time_series,
    get_latent_time_series_sh,
    set_loop_iterations,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FixedPoint:
    """A fixed point found by SCYFI."""

    z: Tensor
    eigenvalues: ndarray
    eigenvectors: ndarray
    classification: str  # 'stable' | 'unstable' | 'saddle'
    region_id: tuple


@dataclass
class Cycle:
    """A k-cycle found by SCYFI."""

    points: list[Tensor]
    period: int
    eigenvalues: ndarray
    classification: str
    region_ids: list[tuple]


# ---------------------------------------------------------------------------
# Helper: duplicate check via rounding
# ---------------------------------------------------------------------------
def _is_duplicate(
    trajectory: list[Tensor],
    existing: list[list[Tensor]],
    digits: int = 3,
) -> bool:
    """Check if any point in trajectory exists in any existing cycle."""
    scale = 10**digits
    for z in trajectory:
        z_q = torch.round(z * scale).long()
        for traj in existing:
            for point in traj:
                p_q = torch.round(point * scale).long()
                if torch.equal(z_q, p_q):
                    return True
    return False


def _is_in_lower_orders(
    z: Tensor,
    found_lower_orders: list[list[list[Tensor]]],
    digits: int = 3,
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


# ---------------------------------------------------------------------------
# Core SCYFI algorithm — PLRNN
# ---------------------------------------------------------------------------
def scy_fi(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    order: int,
    found_lower_orders: list[list[list[Tensor]]],
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
) -> tuple[list[list[Tensor]], list[ndarray]]:
    """Heuristic search for k-cycles in a PLRNN.

    Direct port of ``scy_fi()`` from ``scyfi_algo.jl``.

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
        ``(cycles_found, eigvals)`` — lists of trajectories and
        corresponding eigenvalue arrays.
    """
    dim = A.shape[0]
    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(
        order, outer_loop_iterations, inner_loop_iterations
    )

    i = -1
    while i < outer_loop_iterations:
        i += 1
        relu_matrix_list = construct_relu_matrix_list(dim, order)
        difference_relu_matrices = 1
        c = 0

        while c < inner_loop_iterations:
            c += 1
            z_candidate = get_cycle_point_candidate(
                A, W, relu_matrix_list, h, order
            )

            if z_candidate is not None:
                trajectory = get_latent_time_series(
                    order, A, W, h, dim, z_0=z_candidate
                )

                # Build D matrices from the candidate trajectory
                traj_relu = torch.zeros(order, dim, dim)
                for j in range(order):
                    traj_relu[j] = torch.diag(
                        (trajectory[j] > 0).float()
                    )

                # Check self-consistency
                difference_relu_matrices = 0
                for j in range(order):
                    diff = torch.sum(
                        torch.abs(traj_relu[j] - relu_matrix_list[j])
                    ).item()
                    if diff != 0:
                        difference_relu_matrices = 1
                        break

                    # Check if already found in lower orders
                    if found_lower_orders and _is_in_lower_orders(
                        trajectory[0], found_lower_orders
                    ):
                        difference_relu_matrices = 1
                        break

                if difference_relu_matrices == 0:
                    # Check if not already found at this order
                    if not _is_duplicate(trajectory, cycles_found):
                        e = get_eigvals(A, W, relu_matrix_list, order)
                        cycles_found.append(trajectory)
                        eigvals_found.append(e)
                        i = 0
                        c = 0

                # Update D matrices for next iteration
                if torch.allclose(relu_matrix_list, traj_relu):
                    relu_matrix_list = construct_relu_matrix_list(dim, order)
                else:
                    relu_matrix_list = traj_relu
            else:
                relu_matrix_list = construct_relu_matrix_list(dim, order)

    return cycles_found, eigvals_found


# ---------------------------------------------------------------------------
# Core SCYFI algorithm — shPLRNN
# ---------------------------------------------------------------------------
def scy_fi_sh(
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
    """Heuristic search for k-cycles in a shPLRNN.

    Direct port of the shPLRNN variant of ``scy_fi()`` from
    ``scyfi_algo.jl``.

    Parameters
    ----------
    A : Tensor
        Diagonal entries, shape ``(M,)``.
    W1 : Tensor
        Shape ``(M, H)``.
    W2 : Tensor
        Shape ``(H, M)``.
    h1 : Tensor
        Shape ``(M,)``.
    h2 : Tensor
        Shape ``(H,)``.
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
        ``(cycles_found, eigvals)`` — lists of trajectories and
        corresponding eigenvalue arrays.
    """
    latent_dim = A.shape[0]
    hidden_dim = h2.shape[0]

    cycles_found: list[list[Tensor]] = []
    eigvals_found: list[ndarray] = []

    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(
        order, outer_loop_iterations, inner_loop_iterations
    )

    # Build pool of allowed D matrices
    relu_pool = construct_relu_matrix_pool(
        A, W1, W2, h1, h2, latent_dim, hidden_dim
    )

    i = -1
    while i < outer_loop_iterations:
        i += 1
        relu_matrix_list = construct_relu_matrix_list_from_pool(
            relu_pool, order
        )
        difference_relu_matrices = 1
        c = 0

        while c < inner_loop_iterations:
            c += 1
            z_candidate = get_cycle_point_candidate_sh(
                A, W1, W2, h1, h2, relu_matrix_list, order
            )

            if z_candidate is not None:
                trajectory = get_latent_time_series_sh(
                    order, A, W1, W2, h1, h2, latent_dim, z_0=z_candidate
                )

                # Build D matrices from candidate trajectory
                traj_relu = torch.zeros(order, hidden_dim, hidden_dim)
                for j in range(order):
                    hidden_act = W2 @ trajectory[j] + h2
                    traj_relu[j] = torch.diag((hidden_act > 0).float())

                # Check self-consistency
                difference_relu_matrices = 0
                for j in range(order):
                    diff = torch.sum(
                        torch.abs(traj_relu[j] - relu_matrix_list[j])
                    ).item()
                    if diff != 0:
                        difference_relu_matrices = 1
                        break

                    if found_lower_orders and _is_in_lower_orders(
                        trajectory[0], found_lower_orders
                    ):
                        difference_relu_matrices = 1
                        break

                if difference_relu_matrices == 0:
                    if not _is_duplicate(trajectory, cycles_found):
                        e = get_eigvals_sh(
                            A, W1, W2, relu_matrix_list, order
                        )
                        cycles_found.append(trajectory)
                        eigvals_found.append(e)
                        i = 0
                        c = 0

                if torch.allclose(relu_matrix_list, traj_relu):
                    relu_matrix_list = construct_relu_matrix_list_from_pool(
                        relu_pool, order
                    )
                else:
                    relu_matrix_list = traj_relu
            else:
                relu_matrix_list = construct_relu_matrix_list_from_pool(
                    relu_pool, order
                )

    return cycles_found, eigvals_found


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------
def find_cycles(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    max_order: int,
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
    """Find all cycles up to a given order in a PLRNN.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)`` — diagonal matrix.
    W : Tensor
        Shape ``(M, M)`` — weight matrix.
    h : Tensor
        Shape ``(M,)`` — bias vector.
    max_order : int
        Maximum cycle order to search for.
    outer_loop_iterations : int or None
        Number of random restarts per order.
    inner_loop_iterations : int or None
        Number of inner refinement steps per order.

    Returns
    -------
    tuple[list, list]
        ``(all_cycles, all_eigvals)`` — nested lists indexed by order.
        ``all_cycles[i]`` contains cycles found at order ``i+1``.
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi(
            A,
            W,
            h,
            order,
            all_cycles,
            outer_loop_iterations=outer_loop_iterations,
            inner_loop_iterations=inner_loop_iterations,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals


def find_cycles_sh(
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
    """Find all cycles up to a given order in a shPLRNN.

    Parameters
    ----------
    A : Tensor
        Diagonal entries, shape ``(M,)``.
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
        Number of random restarts per order.
    inner_loop_iterations : int or None
        Number of inner refinement steps per order.

    Returns
    -------
    tuple[list, list]
        ``(all_cycles, all_eigvals)`` indexed by order.
    """
    all_cycles: list[list[list[Tensor]]] = []
    all_eigvals: list[list[ndarray]] = []

    for order in range(1, max_order + 1):
        cycles, eigvals = scy_fi_sh(
            A,
            W1,
            W2,
            h1,
            h2,
            order,
            all_cycles,
            outer_loop_iterations=outer_loop_iterations,
            inner_loop_iterations=inner_loop_iterations,
        )
        all_cycles.append(cycles)
        all_eigvals.append(eigvals)

    return all_cycles, all_eigvals


def find_fixed_points(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    *,
    outer_loop_iterations: int | None = None,
    inner_loop_iterations: int | None = None,
) -> list[FixedPoint]:
    """Find all fixed points in a PLRNN.

    Convenience wrapper over ``find_cycles`` with ``max_order=1``.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    h : Tensor
        Shape ``(M,)``.
    outer_loop_iterations : int or None
        Number of random restarts.
    inner_loop_iterations : int or None
        Number of inner refinement steps.

    Returns
    -------
    list[FixedPoint]
        List of discovered fixed points with eigenstructure.
    """
    from dynamic.analysis.subregions import classify_point, get_region_id

    result = find_cycles(
        A, W, h, 1,
        outer_loop_iterations=outer_loop_iterations,
        inner_loop_iterations=inner_loop_iterations,
    )

    fps: list[FixedPoint] = []
    for idx, traj in enumerate(result[0][0]):
        z = traj[0]
        eigs = result[1][0][idx]
        D = torch.diag((z > 0).float())
        J = (A + W @ D).numpy()
        eigvecs = np.linalg.eig(J)[1]

        fps.append(
            FixedPoint(
                z=z,
                eigenvalues=eigs,
                eigenvectors=eigvecs,
                classification=classify_point(eigs),
                region_id=get_region_id(z),
            )
        )

    return fps
