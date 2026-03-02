"""Bifurcation detection utilities.

Port of ``reference/SCYFI/src/utilities/helpers_bifurcation.jl``.
Detects bifurcations (changes in number or stability of dynamical
objects) along parameter grids and training trajectories.

Note: The Julia version included Plots.jl-based visualisation which
is not ported here. This module provides the data-only detection logic.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


# ---------------------------------------------------------------------------
# Stability comparison
# ---------------------------------------------------------------------------
def compare_stability(
    eigvals: ndarray,
    eigvals_neighbours: list[ndarray],
) -> list[int] | None:
    """Compare stability of one cycle to all neighbouring cycles.

    Compares the total number of stable directions (``|λ| < 1``)
    between ``eigvals`` and each entry in ``eigvals_neighbours``.

    Parameters
    ----------
    eigvals : ndarray
        Eigenvalues of the reference cycle.
    eigvals_neighbours : list[ndarray]
        Eigenvalues of neighbouring cycles.

    Returns
    -------
    list[int] or None
        Indices of neighbours with matching stability pattern,
        or ``None`` if no match found.
    """
    n_stable = int(np.sum(np.abs(eigvals) < 1.0))
    same_indices: list[int] = []

    for i, ev_n in enumerate(eigvals_neighbours):
        n_stable_n = int(np.sum(np.abs(ev_n) < 1.0))
        if n_stable == n_stable_n:
            same_indices.append(i)

    return same_indices if same_indices else None


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------
def get_minimal_state_space_distances(
    cycle: list[Tensor],
    cycles_neighbours: list[list[Tensor]],
) -> list[float]:
    """Minimal Euclidean distance between cycle and each neighbour.

    For each neighbour cycle, compute the distance under all circular
    shifts and keep the minimum.

    Parameters
    ----------
    cycle : list[Tensor]
        Points in the reference cycle.
    cycles_neighbours : list[list[Tensor]]
        List of neighbour cycles.

    Returns
    -------
    list[float]
        One minimal distance per neighbour.
    """
    distances: list[float] = []
    cycle_stacked = torch.stack(cycle)

    for neighbour in cycles_neighbours:
        n_stacked = torch.stack(neighbour)
        best = float("inf")
        n_pts = len(neighbour)
        for shift in range(n_pts):
            # Circular shift
            shifted = torch.roll(n_stacked, shift, dims=0)
            d = torch.norm(cycle_stacked - shifted).item()
            if d < best:
                best = d
        distances.append(best)

    return distances


def get_minimal_eigenvalue_distances(
    eigenvalue: ndarray,
    eigenvalue_neighbours: list[ndarray],
) -> list[float]:
    """Minimal distance between eigenvalue sets.

    For each neighbour eigenvalue set, compute the distance under
    all circular shifts and keep the minimum.

    Parameters
    ----------
    eigenvalue : ndarray
        Eigenvalues of the reference cycle.
    eigenvalue_neighbours : list[ndarray]
        Eigenvalues of neighbouring cycles.

    Returns
    -------
    list[float]
        One minimal distance per neighbour.
    """
    distances: list[float] = []

    for ev_n in eigenvalue_neighbours:
        best = float("inf")
        n_pts = len(ev_n)
        for shift in range(max(1, n_pts)):
            shifted = np.roll(ev_n, shift)
            d = float(np.linalg.norm(eigenvalue - shifted))
            if d < best:
                best = d
        distances.append(best)

    return distances


def get_combined_state_space_eigenvalue_distance(
    cycle: list[Tensor],
    cycles_neighbours: list[list[Tensor]],
    eigenvalue: ndarray,
    eigenvalue_neighbours: list[ndarray],
) -> list[float]:
    """Combined state-space + eigenvalue distances.

    Parameters
    ----------
    cycle : list[Tensor]
        Points in the reference cycle.
    cycles_neighbours : list[list[Tensor]]
        List of neighbour cycles.
    eigenvalue : ndarray
        Eigenvalues of the reference cycle.
    eigenvalue_neighbours : list[ndarray]
        Eigenvalues of neighbouring cycles.

    Returns
    -------
    list[float]
        Sum of state-space and eigenvalue distances per neighbour.
    """
    ss = get_minimal_state_space_distances(cycle, cycles_neighbours)
    ev = get_minimal_eigenvalue_distances(eigenvalue, eigenvalue_neighbours)
    return [s + e for s, e in zip(ss, ev)]


# ---------------------------------------------------------------------------
# Parameter grid construction
# ---------------------------------------------------------------------------
_PARAM_MAP = {
    "a11": ("A", 0, 0),
    "a12": ("A", 0, 1),
    "a21": ("A", 1, 0),
    "a22": ("A", 1, 1),
    "w11": ("W", 0, 0),
    "w12": ("W", 0, 1),
    "w21": ("W", 1, 0),
    "w22": ("W", 1, 1),
    "h1":  ("h", 0),
    "h2":  ("h", 1),
}


def _apply_param_delta(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    param_name: str,
    delta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply a parameter delta to a copy of (A, W, h)."""
    A_new = A.clone()
    W_new = W.clone()
    h_new = h.clone()

    info = _PARAM_MAP.get(param_name)
    if info is None:
        raise ValueError(
            f"Unknown parameter '{param_name}'. "
            f"Valid: {list(_PARAM_MAP.keys())}"
        )

    if info[0] == "A":
        A_new[info[1], info[2]] += delta
    elif info[0] == "W":
        W_new[info[1], info[2]] += delta
    elif info[0] == "h":
        h_new[info[1]] += delta

    return A_new, W_new, h_new


def create_grid_data(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    param_changed_1: str,
    param_changed_2: str,
    param_1: list[float],
    param_2: list[float],
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Prepare parameter variations on a 2D grid.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    h : Tensor
        Shape ``(M,)``.
    param_changed_1 : str
        Name of first varied parameter (e.g., ``'a11'``, ``'w21'``).
    param_changed_2 : str
        Name of second varied parameter.
    param_1 : list[float]
        Delta values for the first parameter.
    param_2 : list[float]
        Delta values for the second parameter.

    Returns
    -------
    list[tuple[Tensor, Tensor, Tensor]]
        List of ``(A_mod, W_mod, h_mod)`` for each grid point.
    """
    grid: list[tuple[Tensor, Tensor, Tensor]] = []
    for p1 in param_1:
        for p2 in param_2:
            A1, W1, h1 = _apply_param_delta(A, W, h, param_changed_1, p1)
            A1, W1, h1 = _apply_param_delta(A1, W1, h1, param_changed_2, p2)
            grid.append((A1, W1, h1))
    return grid


# ---------------------------------------------------------------------------
# Bifurcation detection along a training trajectory
# ---------------------------------------------------------------------------
def find_bifurcations_trajectory(
    cycles_list: list[list[list[Tensor]]],
    eigvals_list: list[list[ndarray]],
    model_numbers: list[int],
) -> list[int]:
    """Detect bifurcations along a training trajectory.

    A bifurcation is detected between consecutive snapshots if:
    1. The number of cycles at any order changes, or
    2. The stability pattern of cycles changes.

    Parameters
    ----------
    cycles_list : list
        ``cycles_list[i]`` = cycles found at snapshot ``i``.
        Each entry is a list of trajectories (one per cycle).
    eigvals_list : list
        ``eigvals_list[i]`` = eigenvalues at snapshot ``i``.
        Each entry is a list of eigenvalue arrays (one per cycle).
    model_numbers : list[int]
        Model/epoch identifiers for each snapshot.

    Returns
    -------
    list[int]
        Model numbers where bifurcations were detected.
    """
    epochs: list[int] = []

    for i in range(len(cycles_list) - 1):
        curr_cycles = cycles_list[i]
        next_cycles = cycles_list[i + 1]
        curr_eigvals = eigvals_list[i]
        next_eigvals = eigvals_list[i + 1]

        # Check number change
        if len(curr_cycles) != len(next_cycles):
            epochs.append(model_numbers[i])
            continue

        # Check stability change
        bifurcation_found = False
        same_stability_indices_found: list[int] = []

        for j in range(len(curr_cycles)):
            if j >= len(curr_eigvals) or not next_eigvals:
                break

            same_stability = compare_stability(
                curr_eigvals[j], next_eigvals
            )

            if same_stability is None:
                epochs.append(model_numbers[i])
                bifurcation_found = True
                break

            if len(same_stability) == 1:
                if same_stability[0] in same_stability_indices_found:
                    epochs.append(model_numbers[i])
                    bifurcation_found = True
                    break
                same_stability_indices_found.extend(same_stability)
            else:
                combined = get_combined_state_space_eigenvalue_distance(
                    curr_cycles[j] if isinstance(curr_cycles[j], list)
                    else [curr_cycles[j]],
                    [next_cycles[k] if isinstance(next_cycles[k], list)
                     else [next_cycles[k]] for k in same_stability],
                    curr_eigvals[j],
                    [next_eigvals[k] for k in same_stability],
                )
                idx_min = int(np.argmin(combined))
                actual_idx = same_stability[idx_min]
                if actual_idx in same_stability_indices_found:
                    epochs.append(model_numbers[i])
                    bifurcation_found = True
                    break
                same_stability_indices_found.append(actual_idx)

        if bifurcation_found:
            continue

    return epochs


# ---------------------------------------------------------------------------
# Bifurcation detection on parameter grid
# ---------------------------------------------------------------------------
def find_bifurcations_parameter_grid(
    cycles_grid: list[list[list[Tensor]]],
    eigvals_grid: list[list[ndarray]],
    coords_grid: list[tuple[float, float]],
    grid_length: int,
) -> list[dict]:
    """Detect bifurcations on a 2D parameter grid.

    A bifurcation edge is placed between adjacent grid points where
    the number or stability of dynamical objects changes.

    Parameters
    ----------
    cycles_grid : list
        ``cycles_grid[i]`` = cycles at grid point ``i``.
    eigvals_grid : list
        ``eigvals_grid[i]`` = eigenvalues at grid point ``i``.
    coords_grid : list[tuple[float, float]]
        ``(param_1, param_2)`` coordinates for each grid point.
    grid_length : int
        Side length of the square grid (total points = grid_length²).

    Returns
    -------
    list[dict]
        List of bifurcation edges, each with:
        ``coord_1``, ``coord_2``, ``type`` (``'number'`` or
        ``'stability'``), and ``direction`` (``'horizontal'`` or
        ``'vertical'``).
    """
    bifurcations: list[dict] = []
    n = grid_length

    for i in range(len(cycles_grid)):
        # Check horizontal neighbour (i+1, same row)
        if (i + 1) % n != 0 and (i + 1) < len(cycles_grid):
            bif = _check_bifurcation_edge(
                cycles_grid[i], eigvals_grid[i],
                cycles_grid[i + 1], eigvals_grid[i + 1],
                coords_grid[i], coords_grid[i + 1],
                "horizontal",
            )
            if bif is not None:
                bifurcations.append(bif)

        # Check vertical neighbour (i+n)
        if i + n < len(cycles_grid):
            bif = _check_bifurcation_edge(
                cycles_grid[i], eigvals_grid[i],
                cycles_grid[i + n], eigvals_grid[i + n],
                coords_grid[i], coords_grid[i + n],
                "vertical",
            )
            if bif is not None:
                bifurcations.append(bif)

    return bifurcations


def _check_bifurcation_edge(
    cycles_a: list[list[Tensor]],
    eigvals_a: list[ndarray],
    cycles_b: list[list[Tensor]],
    eigvals_b: list[ndarray],
    coord_a: tuple[float, float],
    coord_b: tuple[float, float],
    direction: str,
) -> dict | None:
    """Check if a bifurcation occurred between two adjacent grid points."""
    # Number change
    if len(cycles_a) != len(cycles_b):
        return {
            "coord_1": coord_a,
            "coord_2": coord_b,
            "type": "number",
            "direction": direction,
        }

    # Stability change
    same_stability_indices_found: list[int] = []
    for j in range(len(cycles_a)):
        if j >= len(eigvals_a) or not eigvals_b:
            break

        same_stability = compare_stability(eigvals_a[j], eigvals_b)

        if same_stability is None:
            return {
                "coord_1": coord_a,
                "coord_2": coord_b,
                "type": "stability",
                "direction": direction,
            }

        if len(same_stability) == 1:
            if same_stability[0] in same_stability_indices_found:
                return {
                    "coord_1": coord_a,
                    "coord_2": coord_b,
                    "type": "stability",
                    "direction": direction,
                }
            same_stability_indices_found.extend(same_stability)
        else:
            combined = get_combined_state_space_eigenvalue_distance(
                cycles_a[j] if isinstance(cycles_a[j], list)
                else [cycles_a[j]],
                [cycles_b[k] if isinstance(cycles_b[k], list)
                 else [cycles_b[k]] for k in same_stability],
                eigvals_a[j],
                [eigvals_b[k] for k in same_stability],
            )
            idx_min = int(np.argmin(combined))
            actual_idx = same_stability[idx_min]
            if actual_idx in same_stability_indices_found:
                return {
                    "coord_1": coord_a,
                    "coord_2": coord_b,
                    "type": "stability",
                    "direction": direction,
                }
            same_stability_indices_found.append(actual_idx)

    return None
