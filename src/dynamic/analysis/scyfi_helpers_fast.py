"""Optimised SCYFI helper functions.

Performance-focused reimplementations of ``scyfi_helpers.py``.
Key optimisations:

1. **Boolean D-vectors** — ``(order, dim)`` bool instead of ``(order, dim, dim)``
   float matrices.  ``W @ D`` becomes ``W * d`` broadcast (O(dim²) not O(dim³)).
2. **Fused candidate + eigvals** — chain product computed once, reused for both.
3. **Fused simulate + extract** — trajectory generation and D-extraction in one loop.
4. **Hash-based dedup** — O(1) set lookup replaces O(n²) nested loops.
5. **Batched candidates** — ``torch.bmm`` processes multiple random starts in
   parallel.
6. **Pre-cached identity** — ``torch.eye`` allocated once, reused.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


# ---------------------------------------------------------------------------
# D-vector construction  (replaces full diagonal matrices)
# ---------------------------------------------------------------------------
def random_d_vecs(
    dim: int, order: int, *, dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Random binary D-vectors for ``order`` subregions.

    Parameters
    ----------
    dim : int
        Dimensionality.
    order : int
        Number of D-vectors.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    Tensor
        Shape ``(order, dim)`` with entries 0.0 or 1.0.
    """
    return torch.randint(0, 2, (order, dim), dtype=dtype)


def random_d_vecs_batch(
    batch: int, dim: int, order: int, *, dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Batched random D-vectors.

    Returns
    -------
    Tensor
        Shape ``(batch, order, dim)``.
    """
    return torch.randint(0, 2, (batch, order, dim), dtype=dtype)


# ---------------------------------------------------------------------------
# Fused chain product  (z_factor, h_factor, eigvals in one pass)
# ---------------------------------------------------------------------------
def _chain_product(
    A: Tensor, W: Tensor, d_vecs: Tensor, order: int, eye: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute z_factor and h_factor from D-vectors in a single pass.

    z_factor = Π_{i=0}^{order-1} (A + W * d[i])
    h_factor = Σ + I  (accumulated as in reference)

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    d_vecs : Tensor
        Shape ``(order, M)`` — diagonal vectors.
    order : int
        Cycle order.
    eye : Tensor
        Pre-allocated identity ``(M, M)``.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(z_factor, h_factor)`` each ``(M, M)``.
    """
    z_factor = eye.clone()
    h_factor = eye.clone()

    for i in range(order):
        # A + W * d[i] is equivalent to A + W @ diag(d[i])
        AWD = A + W * d_vecs[i]  # (M, M) — broadcast over rows
        z_factor = AWD @ z_factor
        if i > 0:
            h_factor = AWD @ h_factor + eye

    return z_factor, h_factor


def get_candidate_and_eigvals(
    A: Tensor,
    W: Tensor,
    d_vecs: Tensor,
    h: Tensor,
    order: int,
    eye: Tensor,
) -> tuple[Tensor | None, ndarray]:
    """Fused cycle-point candidate and eigenvalue computation.

    Computes the chain product once, then derives both the candidate
    ``z* = (I - z_factor)^{-1} h_factor h`` and eigenvalues of
    ``z_factor``.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    d_vecs : Tensor
        Shape ``(order, M)``.
    h : Tensor
        Shape ``(M,)``.
    order : int
        Cycle order.
    eye : Tensor
        Pre-allocated identity ``(M, M)``.

    Returns
    -------
    tuple[Tensor | None, ndarray]
        ``(candidate, eigenvalues)``.  candidate is None if singular.
    """
    z_factor, h_factor = _chain_product(A, W, d_vecs, order, eye)

    eigvals = np.linalg.eigvals(z_factor.numpy())

    try:
        inv_mat = torch.linalg.inv(eye - z_factor)
        candidate = inv_mat @ (h_factor @ h)
        return candidate, eigvals
    except torch.linalg.LinAlgError:
        return None, eigvals


# ---------------------------------------------------------------------------
# Batched chain product with bmm
# ---------------------------------------------------------------------------
def batch_candidates(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    d_vecs_batch: Tensor,
    order: int,
    eye: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute candidates for a batch of D-vector sequences in parallel.

    Parameters
    ----------
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    h : Tensor
        Shape ``(M,)``.
    d_vecs_batch : Tensor
        Shape ``(B, order, M)``.
    order : int
        Cycle order.
    eye : Tensor
        Pre-allocated identity ``(M, M)``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        - ``candidates`` shape ``(B, M)``
        - ``z_factors`` shape ``(B, M, M)`` — for eigenvalue computation
        - ``valid`` shape ``(B,)`` bool — True where inversion succeeded
    """
    B = d_vecs_batch.shape[0]
    dim = A.shape[0]

    z_factors = eye.unsqueeze(0).expand(B, -1, -1).clone()
    h_factors = eye.unsqueeze(0).expand(B, -1, -1).clone()
    eye_b = eye.unsqueeze(0)  # (1, M, M) for broadcasting

    for i in range(order):
        # d_vecs_batch[:, i, :] is (B, M)
        # W * d  broadcasted: (M, M) * (B, 1, M) → (B, M, M)
        WD = W.unsqueeze(0) * d_vecs_batch[:, i, :].unsqueeze(1)
        AWD = A.unsqueeze(0) + WD  # (B, M, M)
        z_factors = torch.bmm(AWD, z_factors)
        if i > 0:
            h_factors = torch.bmm(AWD, h_factors) + eye_b

    # Batched inverse
    I_minus_z = eye.unsqueeze(0).expand(B, -1, -1) - z_factors

    # Use torch.linalg.inv with error handling per batch item
    candidates = torch.zeros(B, dim, dtype=A.dtype)
    valid = torch.zeros(B, dtype=torch.bool)

    # Try batched inverse first (fast path — works if all are invertible)
    try:
        inv_batch = torch.linalg.inv(I_minus_z)
        rhs = torch.bmm(
            h_factors, h.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        )  # (B, M, 1)
        candidates = torch.bmm(inv_batch, rhs).squeeze(-1)  # (B, M)
        valid[:] = True
    except torch.linalg.LinAlgError:
        # Fall back to per-item inversion
        for b in range(B):
            try:
                inv_b = torch.linalg.inv(I_minus_z[b])
                candidates[b] = inv_b @ (h_factors[b] @ h)
                valid[b] = True
            except torch.linalg.LinAlgError:
                pass

    return candidates, z_factors, valid


# ---------------------------------------------------------------------------
# Fused simulate + D-extraction
# ---------------------------------------------------------------------------
def simulate_and_extract(
    z0: Tensor, A: Tensor, W: Tensor, h: Tensor, order: int,
) -> tuple[Tensor, Tensor]:
    """Simulate trajectory AND extract D-vectors in one pass.

    Parameters
    ----------
    z0 : Tensor
        Initial state ``(M,)``.
    A : Tensor
        Shape ``(M, M)``.
    W : Tensor
        Shape ``(M, M)``.
    h : Tensor
        Shape ``(M,)``.
    order : int
        Number of steps.

    Returns
    -------
    tuple[Tensor, Tensor]
        - ``trajectory`` shape ``(order, M)``
        - ``d_vecs`` shape ``(order, M)`` with 0.0/1.0 entries
    """
    dim = A.shape[0]
    traj = torch.empty(order, dim, dtype=A.dtype)
    d_vecs = torch.empty(order, dim, dtype=A.dtype)

    z = z0
    for i in range(order):
        traj[i] = z
        d_vecs[i] = (z > 0).to(A.dtype)
        z = A @ z + W @ torch.clamp(z, min=0.0) + h

    return traj, d_vecs


def simulate_and_extract_sh(
    z0: Tensor,
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    h1: Tensor,
    h2: Tensor,
    order: int,
) -> tuple[Tensor, Tensor]:
    """Simulate shPLRNN trajectory AND extract D-vectors in one pass.

    Parameters
    ----------
    z0 : Tensor
        Initial state ``(M,)``.
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
        Number of steps.

    Returns
    -------
    tuple[Tensor, Tensor]
        - ``trajectory`` shape ``(order, M)``
        - ``d_vecs`` shape ``(order, H)`` with 0.0/1.0 entries
    """
    latent_dim = A.shape[0]
    hidden_dim = h2.shape[0]
    traj = torch.empty(order, latent_dim, dtype=A.dtype)
    d_vecs = torch.empty(order, hidden_dim, dtype=A.dtype)

    z = z0
    for i in range(order):
        traj[i] = z
        hidden_act = W2 @ z + h2
        d_vecs[i] = (hidden_act > 0).to(A.dtype)
        z = A * z + W1 @ torch.clamp(hidden_act, min=0.0) + h1

    return traj, d_vecs


# ---------------------------------------------------------------------------
# Hash-based deduplication
# ---------------------------------------------------------------------------
def make_key(z: Tensor, scale: int = 1000) -> tuple[int, ...]:
    """Quantise a state vector into a hashable key.

    Parameters
    ----------
    z : Tensor
        State vector ``(M,)``.
    scale : int
        Rounding scale (10^digits).

    Returns
    -------
    tuple[int, ...]
        Hashable key.
    """
    return tuple(torch.round(z * scale).long().tolist())


def make_key_from_row(z_row: Tensor, scale: int = 1000) -> tuple[int, ...]:
    """Same as make_key but for a row of a 2D tensor."""
    return tuple(torch.round(z_row * scale).long().tolist())


# ---------------------------------------------------------------------------
# shPLRNN chain product
# ---------------------------------------------------------------------------
def _chain_product_sh(
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    d_vecs: Tensor,
    order: int,
    eye: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute z/h1/h2 factors for shPLRNN from D-vectors.

    Parameters
    ----------
    A : Tensor
        Diagonal entries ``(M,)``.
    W1 : Tensor
        Shape ``(M, H)``.
    W2 : Tensor
        Shape ``(H, M)``.
    d_vecs : Tensor
        Shape ``(order, H)``.
    order : int
        Cycle order.
    eye : Tensor
        Pre-allocated identity ``(M, M)``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(z_factor, h1_factor, h2_factor)``.
    """
    A_diag = torch.diag(A)

    z_factor = eye.clone()
    h1_factor = eye.clone()
    # W1 * d[0] is column-wise masking of W1
    h2_factor = (W1 * d_vecs[0]).clone()

    for i in range(order - 1):
        # m = diag(A) + W1 * d[i] @ W2  — but d[i] is a vector
        # W1 * d[i] masks columns, then @ W2
        W1D = W1 * d_vecs[i]  # (M, H) — column mask
        m = A_diag + W1D @ W2  # (M, M)
        z_factor = m @ z_factor

        W1D_next = W1 * d_vecs[i + 1]
        m_next = A_diag + W1D_next @ W2
        h1_factor = m_next @ h1_factor + eye
        h2_factor = m_next @ h2_factor + W1D_next

    # Final z_factor step
    W1D_last = W1 * d_vecs[order - 1]
    m_last = A_diag + W1D_last @ W2
    z_factor = m_last @ z_factor

    return z_factor, h1_factor, h2_factor


def get_candidate_and_eigvals_sh(
    A: Tensor,
    W1: Tensor,
    W2: Tensor,
    h1: Tensor,
    h2: Tensor,
    d_vecs: Tensor,
    order: int,
    eye: Tensor,
) -> tuple[Tensor | None, ndarray]:
    """Fused shPLRNN candidate + eigenvalue computation.

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
    d_vecs : Tensor
        Shape ``(order, H)``.
    order : int
        Cycle order.
    eye : Tensor
        Pre-allocated identity ``(M, M)``.

    Returns
    -------
    tuple[Tensor | None, ndarray]
        ``(candidate, eigenvalues)``.
    """
    z_factor, h1_factor, h2_factor = _chain_product_sh(
        A, W1, W2, d_vecs, order, eye,
    )

    eigvals = np.linalg.eigvals(z_factor.numpy())

    try:
        inv_mat = torch.linalg.inv(eye - z_factor)
        candidate = inv_mat @ (h1_factor @ h1 + h2_factor @ h2)
        return candidate, eigvals
    except torch.linalg.LinAlgError:
        return None, eigvals
