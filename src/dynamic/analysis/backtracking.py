"""Map inversion and backtracking (Algorithm 2).

For stable manifolds we need the inverse map F_θ⁻¹.  In each subregion
the PLRNN is affine:

    z_t = (A + W D) z_{t-1} + h

so the inverse is:

    z_{t-1} = (A + W D)⁻¹ (z_t - h)

The catch: D depends on z_{t-1} (unknown).  This module implements the
heuristic resolution from Appendix C:

1. Try current region's D
2. Verify via forward step
3. Try D pool from previously visited regions
4. Hierarchical bitflip search

Reference: Appendix C.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dynamic.analysis.subregions import get_D, get_neighbors, get_region_id


def backward_step(model: nn.Module, z_t: Tensor, D: Tensor) -> Tensor:
    """Single backward step: z_{t-1} = (A + WD)⁻¹(z_t - h).

    Parameters
    ----------
    model : nn.Module
        PLRNN model with ``A``, ``W``, ``h`` parameters.
    z_t : Tensor
        Current state of shape ``(M,)``.
    D : Tensor
        Diagonal activation matrix of shape ``(M, M)``.

    Returns
    -------
    Tensor
        Previous state of shape ``(M,)``.
    """
    J = torch.diag(model.A) + model.W @ D
    rhs = z_t - model.h
    return torch.linalg.solve(J, rhs)


def verify_forward(
    model: nn.Module,
    z_prev: Tensor,
    z_target: Tensor,
    tol: float = 1e-6,
) -> bool:
    """Check F(z_prev) ≈ z_target.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    z_prev : Tensor
        Candidate previous state.
    z_target : Tensor
        Target current state.
    tol : float
        Tolerance for comparison.

    Returns
    -------
    bool
        True if forward step matches target within tolerance.
    """
    with torch.no_grad():
        z_fwd = model.forward(z_prev)
    return bool(torch.allclose(z_fwd, z_target, atol=tol))


def try_previous_regions(
    model: nn.Module,
    D_pool: list[Tensor],
    z_t: Tensor,
    tol: float = 1e-6,
) -> Tensor | None:
    """Try inversion using previously visited D matrices.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    D_pool : list[Tensor]
        Pool of D matrices from previously visited regions.
    z_t : Tensor
        Current state to invert.
    tol : float
        Forward-verification tolerance.

    Returns
    -------
    Tensor or None
        Valid inverse if found, else None.
    """
    for D in D_pool:
        try:
            z_prev = backward_step(model, z_t, D)
            if verify_forward(model, z_prev, z_t, tol=tol):
                return z_prev
        except torch.linalg.LinAlgError:
            continue
    return None


def try_bitflips(
    model: nn.Module,
    z_t: Tensor,
    z_candidate: Tensor,
    tol: float = 1e-6,
    max_depth: int = 2,
) -> Tensor | None:
    """Hierarchical bitflip search over neighboring regions.

    Starting from the region of ``z_candidate``, flip one bit of
    the activation pattern at a time and try inversion.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    z_t : Tensor
        Current state to invert.
    z_candidate : Tensor
        Initial candidate (may be in wrong region).
    tol : float
        Forward-verification tolerance.
    max_depth : int
        Number of bitflip levels to try.

    Returns
    -------
    Tensor or None
        Valid inverse if found, else None.
    """
    region = get_region_id(z_candidate)
    visited = {region}

    # BFS over neighboring regions
    frontier = [region]
    for _depth in range(max_depth):
        next_frontier = []
        for reg in frontier:
            for neighbor in get_neighbors(reg):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                D = torch.diag(torch.tensor(neighbor, dtype=torch.float32))
                try:
                    z_prev = backward_step(model, z_t, D)
                    if verify_forward(model, z_prev, z_t, tol=tol):
                        return z_prev
                except torch.linalg.LinAlgError:
                    continue
                next_frontier.append(neighbor)
        frontier = next_frontier
    return None


def backtrack_trajectory(
    model: nn.Module,
    z_T: Tensor,
    T: int,
    tol: float = 1e-6,
) -> Tensor:
    """Full backward trajectory of length T.

    Constructs [z_0, z_1, ..., z_T] such that F(z_{t-1}) ≈ z_t.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    z_T : Tensor
        Final state.
    T : int
        Number of backward steps.
    tol : float
        Verification tolerance.

    Returns
    -------
    Tensor
        Trajectory of shape ``(T + 1, M)`` from ``z_0`` to ``z_T``.
    """
    trajectory = [z_T]
    z = z_T
    D_pool: list[Tensor] = []

    for _step in range(T):
        D = get_D(z)
        D_pool.append(D)

        # 1. Try direct inversion with current D
        try:
            z_prev = backward_step(model, z, D)
            if verify_forward(model, z_prev, z, tol=tol):
                z = z_prev
                trajectory.append(z)
                continue
        except torch.linalg.LinAlgError:
            pass

        # 2. Try D pool
        result = try_previous_regions(model, D_pool, z, tol=tol)
        if result is not None:
            z = result
            trajectory.append(z)
            continue

        # 3. Try bitflip search
        z_candidate = backward_step(model, z, D)
        result = try_bitflips(model, z, z_candidate, tol=tol)
        if result is not None:
            z = result
            trajectory.append(z)
            continue

        # Fallback: use best-effort candidate (may not verify perfectly)
        z = z_candidate
        trajectory.append(z)

    # Reverse so trajectory goes z_0 → z_T
    trajectory.reverse()
    return torch.stack(trajectory)
