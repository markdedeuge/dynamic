"""SCYFI core helper functions.

Direct port of ``reference/SCYFI/src/utilities/helpers.jl``.
Low-level mathematical operations for the SCYFI algorithm:
ReLU matrix construction, factor computation, candidate solving,
time series generation, and eigenvalue extraction.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


# ---------------------------------------------------------------------------
# ReLU matrix construction
# ---------------------------------------------------------------------------
def construct_relu_matrix(
    quadrant_number: int, dim: int, *, dtype=None
) -> Tensor:
    """Single diagonal D matrix from an integer quadrant index.

    The binary representation of ``quadrant_number`` encodes which
    diagonal entries are 1 (active) vs 0 (inactive).

    Parameters
    ----------
    quadrant_number : int
        Integer in ``[0, 2^dim)``.
    dim : int
        Dimensionality.
    dtype : torch.dtype, optional
        Output dtype (default: torch default).

    Returns
    -------
    Tensor
        Diagonal matrix of shape ``(dim, dim)``.
    """
    bits = format(quadrant_number, f"0{dim}b")
    diag = torch.tensor([float(b) for b in reversed(bits)], dtype=dtype)
    return torch.diag(diag)


def construct_relu_matrix_list(
    dim: int, order: int, *, dtype=None
) -> Tensor:
    """Random sequence of D matrices for ``order`` subregions.

    Parameters
    ----------
    dim : int
        Dimensionality.
    order : int
        Number of D matrices to generate.
    dtype : torch.dtype, optional
        Output dtype (default: torch default).

    Returns
    -------
    Tensor
        Shape ``(order, dim, dim)``.
    """
    D_list = torch.zeros(order, dim, dim, dtype=dtype)
    for i in range(order):
        n = int(torch.randint(0, 2**dim, (1,)).item())
        D_list[i] = construct_relu_matrix(n, dim, dtype=dtype)
    return D_list


def construct_relu_matrix_list_from_pool(
    pool: Tensor, order: int
) -> Tensor:
    """Random D sequence drawn from an allowed pool (shPLRNN).

    Parameters
    ----------
    pool : Tensor
        Shape ``(N, H, H)`` pool of allowed D matrices.
    order : int
        Number of D matrices to draw.

    Returns
    -------
    Tensor
        Shape ``(order, H, H)``.
    """
    indices = torch.randint(0, pool.shape[0], (order,))
    return pool[indices]


def construct_relu_matrix_pool(
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    h1: Tensor,
    h2: Tensor,
    dim: int,
    hidden_dim: int,
    n_points: int = 1_000_000,
) -> Tensor:
    """Monte Carlo pool of reachable activation patterns for shPLRNN.

    Samples random z vectors and computes which hidden-layer activation
    patterns ``(W₂ z + h₂) > 0`` are reachable, returning the unique
    set of diagonal D matrices.

    Parameters
    ----------
    A : Tensor
        Diagonal entries, shape ``(dim,)``.
    W1 : Tensor
        Shape ``(dim, hidden_dim)``.
    W2 : Tensor
        Shape ``(hidden_dim, dim)``.
    h1 : Tensor
        Shape ``(dim,)``.
    h2 : Tensor
        Shape ``(hidden_dim,)``.
    dim : int
        Latent dimensionality.
    hidden_dim : int
        Hidden dimensionality.
    n_points : int
        Number of random points to sample.

    Returns
    -------
    Tensor
        Shape ``(N, hidden_dim, hidden_dim)`` of unique diagonal D's.
    """
    # Sample random z vectors uniformly in [-10, 10]
    m = torch.rand(dim, n_points, dtype=W2.dtype) * 20.0 - 10.0

    # Compute activation patterns: (W2 @ z + h2) > 0 for each column
    activations = W2 @ m + h2.unsqueeze(1)  # (H, n_points)
    patterns = (activations > 0).int().T  # (n_points, H)

    # Find unique patterns
    unique_patterns = torch.unique(patterns, dim=0)

    # Build diagonal matrices
    n_unique = unique_patterns.shape[0]
    D_list = torch.zeros(n_unique, hidden_dim, hidden_dim, dtype=W2.dtype)
    for k in range(n_unique):
        D_list[k] = torch.diag(unique_patterns[k].to(W2.dtype))

    return D_list


# ---------------------------------------------------------------------------
# Factor computation (PLRNN)
# ---------------------------------------------------------------------------
def get_factor_in_front_of_z(
    A: Tensor, W: Tensor, D_list: Tensor, order: int
) -> Tensor:
    """Compute the z-factor: ``Π_{i=1}^{order} (A + W·Dᵢ)``.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    D_list : Tensor
        Shape ``(order, M, M)``.
    order : int
        Cycle order.

    Returns
    -------
    Tensor
        Product matrix, shape ``(M, M)``.
    """
    dim = A.shape[0]
    factor = torch.eye(dim, dtype=A.dtype)
    for i in range(order):
        factor = (A + W @ D_list[i]) @ factor
    return factor


def get_factor_in_front_of_h(
    A: Tensor, W: Tensor, D_list: Tensor, order: int
) -> Tensor:
    """Compute the h-factor for the cycle equation.

    For order 1, returns identity.
    For order k, computes: ``Σ_{i=2}^{k} Π_{j=i}^{k} (A+W·Dⱼ) + I``.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    D_list : Tensor
        Shape ``(order, M, M)`` — full D_list (uses indices 1..end).
    order : int
        Cycle order.

    Returns
    -------
    Tensor
        Factor matrix, shape ``(M, M)``.
    """
    dim = A.shape[0]
    factor = torch.eye(dim, dtype=A.dtype)
    # Julia: iterates over D_list[:,:,2:end] for order-1 steps
    for i in range(1, order):
        factor = (A + W @ D_list[i]) @ factor + torch.eye(dim, dtype=A.dtype)
    return factor


# ---------------------------------------------------------------------------
# Factor computation (shPLRNN)
# ---------------------------------------------------------------------------
def get_factors_sh(
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    D_list: Tensor,
    order: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute z/h₁/h₂ factors for shPLRNN cycle equation.

    Parameters
    ----------
    A : Tensor
        Diagonal entries, shape ``(M,)``.
    W1 : Tensor
        Shape ``(M, H)``.
    W2 : Tensor
        Shape ``(H, M)``.
    D_list : Tensor
        Shape ``(order, H, H)``.
    order : int
        Cycle order.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(z_factor, h1_factor, h2_factor)`` each of shape ``(M, M)``
        or ``(M, H)`` for h2_factor.
    """
    latent_dim = W1.shape[0]

    factor_z = torch.eye(latent_dim, dtype=A.dtype)
    factor_h1 = torch.eye(latent_dim, dtype=A.dtype)
    factor_h2 = (W1 @ D_list[0]).clone()

    A_diag = torch.diag(A)

    for i in range(1, order):
        m = A_diag + W1 @ D_list[i] @ W2
        if i < order:
            factor_z = m @ factor_z
        factor_h1 = m @ factor_h1 + torch.eye(latent_dim, dtype=A.dtype)
        factor_h2 = m @ factor_h2 + W1 @ D_list[i]

    # Final z-factor step (the last one in the Julia loop)
    if order > 0:
        m_last = A_diag + W1 @ D_list[order - 1] @ W2
        # The Julia code applies the last factor separately after the loop:
        # factor_z = (Diagonal(A) + (W₁*D_list[:,:,order])*W₂)*factor_z
        # But we already applied it in the loop. We need to match Julia exactly.
        # Let's re-do to match the Julia logic precisely.
        pass

    # Re-implement to match Julia exactly
    factor_z = torch.eye(latent_dim, dtype=A.dtype)
    factor_h1 = torch.eye(latent_dim, dtype=A.dtype)
    factor_h2 = (W1 @ D_list[0]).clone()

    for i in range(order - 1):
        m_i = A_diag + W1 @ D_list[i] @ W2
        factor_z = m_i @ factor_z
        m_next = A_diag + W1 @ D_list[i + 1] @ W2
        factor_h1 = m_next @ factor_h1 + torch.eye(latent_dim, dtype=A.dtype)
        factor_h2 = m_next @ factor_h2 + W1 @ D_list[i + 1]

    # Final z-factor application
    m_last = A_diag + W1 @ D_list[order - 1] @ W2
    factor_z = m_last @ factor_z

    return factor_z, factor_h1, factor_h2


# ---------------------------------------------------------------------------
# Cycle point candidates
# ---------------------------------------------------------------------------
def get_cycle_point_candidate(
    A: Tensor,
    W: Tensor,
    D_list: Tensor,
    h: Tensor,
    order: int,
) -> Tensor | None:
    """Solve the cycle equation for a candidate fixed/cycle point.

    Computes ``z* = (I - z_factor)⁻¹ · h_factor · h``.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    D_list : Tensor
        Shape ``(order, M, M)``.
    h : Tensor
        Shape ``(M,)``.
    order : int
        Cycle order.

    Returns
    -------
    Tensor or None
        Candidate point of shape ``(M,)``, or None if the system
        is singular.
    """
    z_factor = get_factor_in_front_of_z(A, W, D_list, order)
    dim = A.shape[0]
    try:
        inverse_matrix = torch.linalg.inv(
            torch.eye(dim, dtype=A.dtype) - z_factor
        )
        h_factor = get_factor_in_front_of_h(A, W, D_list, order)
        z_candidate = inverse_matrix @ (h_factor @ h)
        return z_candidate
    except torch.linalg.LinAlgError:
        return None


def get_cycle_point_candidate_sh(
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    h1: Tensor,
    h2: Tensor,
    D_list: Tensor,
    order: int,
) -> Tensor | None:
    """Solve the shPLRNN cycle equation for a candidate point.

    Computes ``z* = (I - z_factor)⁻¹ · (h₁_factor·h₁ + h₂_factor·h₂)``.

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
    D_list : Tensor
        Shape ``(order, H, H)``.
    order : int
        Cycle order.

    Returns
    -------
    Tensor or None
        Candidate point of shape ``(M,)``, or None if singular.
    """
    z_factor, h1_factor, h2_factor = get_factors_sh(A, W1, W2, D_list, order)
    latent_dim = W1.shape[0]
    try:
        inverse_matrix = torch.linalg.inv(
            torch.eye(latent_dim, dtype=A.dtype) - z_factor
        )
        z_candidate = inverse_matrix @ (h1_factor @ h1 + h2_factor @ h2)
        return z_candidate
    except torch.linalg.LinAlgError:
        return None


# ---------------------------------------------------------------------------
# Latent step and time series
# ---------------------------------------------------------------------------
def latent_step(z: Tensor, A: Tensor, W: Tensor, h: Tensor) -> Tensor:
    """Single PLRNN step: ``A z + W max(0, z) + h``.

    Parameters
    ----------
    z : Tensor
        State vector, shape ``(M,)``.
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    h : Tensor
        Shape ``(M,)``.

    Returns
    -------
    Tensor
        Next state, shape ``(M,)``.
    """
    return A @ z + W @ torch.clamp(z, min=0.0) + h


def latent_step_sh(
    z: Tensor,
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    h1: Tensor,
    h2: Tensor,
) -> Tensor:
    """Single shPLRNN step: ``A .* z + W₁ max(0, W₂ z + h₂) + h₁``.

    Parameters
    ----------
    z : Tensor
        State vector, shape ``(M,)``.
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

    Returns
    -------
    Tensor
        Next state, shape ``(M,)``.
    """
    return A * z + W1 @ torch.clamp(W2 @ z + h2, min=0.0) + h1


def get_latent_time_series(
    time_steps: int,
    A: Tensor,
    W: Tensor,
    h: Tensor,
    dz: int,
    *,
    z_0: Tensor | None = None,
) -> list[Tensor]:
    """Generate a time series by iteratively applying the PLRNN.

    Parameters
    ----------
    time_steps : int
        Number of states to produce.
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    h : Tensor
        Shape ``(M,)``.
    dz : int
        Dimensionality (used if z_0 is None).
    z_0 : Tensor or None
        Initial state. If None, random initialisation.

    Returns
    -------
    list[Tensor]
        List of ``time_steps`` state tensors.
    """
    if z_0 is None:
        z = torch.randn(dz)
    else:
        z = z_0.clone()

    trajectory: list[Tensor] = [z]
    for _ in range(time_steps - 1):
        z = latent_step(z, A, W, h)
        trajectory.append(z)
    return trajectory


def get_latent_time_series_sh(
    time_steps: int,
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    h1: Tensor,
    h2: Tensor,
    dz: int,
    *,
    z_0: Tensor | None = None,
) -> list[Tensor]:
    """Generate a time series by iteratively applying the shPLRNN.

    Parameters
    ----------
    time_steps : int
        Number of states to produce.
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
    dz : int
        Dimensionality (used if z_0 is None).
    z_0 : Tensor or None
        Initial state.

    Returns
    -------
    list[Tensor]
        List of ``time_steps`` state tensors.
    """
    if z_0 is None:
        z = torch.randn(dz)
    else:
        z = z_0.clone()

    trajectory: list[Tensor] = [z]
    for _ in range(time_steps - 1):
        z = latent_step_sh(z, A, W1, W2, h1, h2)
        trajectory.append(z)
    return trajectory


# ---------------------------------------------------------------------------
# Eigenvalue computation
# ---------------------------------------------------------------------------
def get_eigvals(
    A: Tensor, W: Tensor, D_list: Tensor, order: int
) -> ndarray:
    """Eigenvalues of the composed Jacobian for stability analysis.

    Computes ``eigvals(Π_{i=1}^{order} (A + W·Dᵢ))``.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    D_list : Tensor
        Shape ``(order, M, M)``.
    order : int
        Cycle order.

    Returns
    -------
    ndarray
        Array of eigenvalues.
    """
    dim = A.shape[0]
    e = torch.eye(dim, dtype=A.dtype)
    for i in range(order):
        e = (A + W @ D_list[i]) @ e
    return np.linalg.eigvals(e.numpy())


def get_eigvals_sh(
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    D_list: Tensor,
    order: int,
) -> ndarray:
    """Eigenvalues of the composed shPLRNN Jacobian.

    Computes ``eigvals(Π_{i=1}^{order} (diag(A) + W₁·Dᵢ·W₂))``.

    Parameters
    ----------
    A : Tensor
        Diagonal entries, shape ``(M,)``.
    W1 : Tensor
        Shape ``(M, H)``.
    W2 : Tensor
        Shape ``(H, M)``.
    D_list : Tensor
        Shape ``(order, H, H)``.
    order : int
        Cycle order.

    Returns
    -------
    ndarray
        Array of eigenvalues.
    """
    latent_dim = W1.shape[0]
    e = torch.eye(latent_dim, dtype=A.dtype)
    A_diag = torch.diag(A)
    for i in range(order):
        e = (A_diag + W1 @ D_list[i] @ W2) @ e
    return np.linalg.eigvals(e.numpy())


# ---------------------------------------------------------------------------
# Loop iteration defaults
# ---------------------------------------------------------------------------
def set_loop_iterations(
    order: int,
    outer_loop: int | None,
    inner_loop: int | None,
) -> tuple[int, int]:
    """Set default hyperparameters if not provided.

    Defaults are tuned based on the Julia reference implementation.

    Parameters
    ----------
    order : int
        Cycle order being searched.
    outer_loop : int or None
        Outer loop iterations (None for default).
    inner_loop : int or None
        Inner loop iterations (None for default).

    Returns
    -------
    tuple[int, int]
        ``(outer_loop_iterations, inner_loop_iterations)``.
    """
    if outer_loop is None:
        if order < 8:
            outer_loop = 10
        elif order < 30:
            outer_loop = 40
        else:
            outer_loop = 100

    if inner_loop is None:
        if order < 3:
            inner_loop = 20
        elif order < 6:
            inner_loop = 60
        elif order < 8:
            inner_loop = 300
        elif order < 20:
            inner_loop = 1080
        else:
            inner_loop = 1115

    return outer_loop, inner_loop
