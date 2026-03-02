"""Subregion utilities for piecewise-linear RNNs.

Model-agnostic functions for working with the linear subregions
induced by ReLU activations in PLRNNs.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def get_D(z: Tensor) -> Tensor:
    """Diagonal ReLU activation matrix.

    Parameters
    ----------
    z : Tensor
        State vector of shape ``(M,)``.

    Returns
    -------
    Tensor
        Diagonal matrix of shape ``(M, M)`` with ``D[i,i] = 1`` if
        ``z[i] > 0``, else ``0``.
    """
    return torch.diag((z > 0).float())


def get_region_id(z: Tensor) -> tuple[int, ...]:
    """Hashable subregion identifier from sign pattern.

    Parameters
    ----------
    z : Tensor
        State vector of shape ``(M,)``.

    Returns
    -------
    tuple[int, ...]
        Binary tuple of length ``M`` where entry ``i`` is ``1`` if
        ``z[i] > 0``, else ``0``.
    """
    return tuple(int(x) for x in (z > 0).tolist())


def get_neighbors(region_id: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Adjacent regions reachable by a single bit-flip in *D*.

    Parameters
    ----------
    region_id : tuple[int, ...]
        Binary tuple describing the current subregion.

    Returns
    -------
    list[tuple[int, ...]]
        List of ``M`` neighbor regions, each differing by exactly one bit.
    """
    neighbors: list[tuple[int, ...]] = []
    for i in range(len(region_id)):
        flipped = list(region_id)
        flipped[i] = 1 - flipped[i]
        neighbors.append(tuple(flipped))
    return neighbors


def get_jacobian_in_region(
    A: Tensor,
    W: Tensor,
    D: Tensor,
) -> Tensor:
    """Compute the Jacobian ``A + W·D`` for a given activation pattern.

    Parameters
    ----------
    A : Tensor
        Diagonal (or full) matrix of shape ``(M, M)``.
    W : Tensor
        Weight matrix of shape ``(M, M)``.
    D : Tensor
        Diagonal activation matrix of shape ``(M, M)``.

    Returns
    -------
    Tensor
        Jacobian matrix of shape ``(M, M)``.
    """
    return A + W @ D


def classify_point(eigenvalues: ndarray) -> str:
    """Classify a fixed point or cycle by eigenvalue magnitudes.

    Parameters
    ----------
    eigenvalues : ndarray
        Array of (possibly complex) eigenvalues.

    Returns
    -------
    str
        ``'stable'`` if all ``|λ| < 1``,
        ``'unstable'`` if all ``|λ| > 1``,
        ``'saddle'`` otherwise (mixed magnitudes).
    """
    magnitudes = np.abs(eigenvalues)
    all_stable = bool(np.all(magnitudes < 1.0))
    all_unstable = bool(np.all(magnitudes > 1.0))

    if all_stable:
        return "stable"
    if all_unstable:
        return "unstable"
    return "saddle"
