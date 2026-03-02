"""Manifold quality metric δ_σ (Eq 6).

Quantifies how well a point lies on the manifold by measuring
its convergence to the saddle point under forward (unstable) or
backward (stable) iteration.

Reference: §3.3, Eq 6.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def delta_sigma(
    x0: Tensor,
    model: nn.Module,
    saddle: Tensor,
    sigma: int,
    k_max: int = 1000,
) -> float:
    """Manifold convergence metric (Eq 6).

    δ_σ(x0) = min_{kσ ≥ 0} ‖F^k(x0) - p‖² / ‖x0 - p‖²

    For stable manifolds (σ=+1), iterate forward and track the
    minimum distance ratio. For unstable manifolds (σ=-1), iterate
    forward as well (the point should converge to p under F).

    Parameters
    ----------
    x0 : Tensor
        Test point of shape ``(M,)``.
    model : nn.Module
        PLRNN model.
    saddle : Tensor
        Saddle fixed point of shape ``(M,)``.
    sigma : int
        +1 for stable, -1 for unstable manifold.
    k_max : int
        Maximum number of iterations.

    Returns
    -------
    float
        δ_σ value in [0, 1]. Near 0 indicates on-manifold.
    """
    d0_sq = torch.sum((x0 - saddle) ** 2).item()
    if d0_sq < 1e-20:
        return 0.0

    z = x0.clone().detach()
    min_ratio = 1.0

    with torch.no_grad():
        for _k in range(k_max):
            z = model.forward(z)
            dk_sq = torch.sum((z - saddle) ** 2).item()
            ratio = dk_sq / d0_sq
            ratio = min(ratio, 1.0)  # clamp to [0, 1]
            if ratio < min_ratio:
                min_ratio = ratio
            if min_ratio < 1e-12:
                return 0.0

    return max(0.0, min(min_ratio, 1.0))


def delta_sigma_statistic(
    model: nn.Module,
    saddle: Tensor,
    sigma: int,
    U_min: Tensor,
    U_max: Tensor,
    N_samples: int = 1000,
    k_max: int = 500,
    manifold_points: Tensor | None = None,
) -> float:
    """Full Δ_σ quality statistic.

    Δ_σ = ⟨I_U⟩ - median(δ_σ for on-manifold points in U)

    where I_U is the indicator for random points with δ_σ exceeding
    the maximum on-manifold δ_σ.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    saddle : Tensor
        Saddle fixed point.
    sigma : int
        +1 for stable, -1 for unstable.
    U_min : Tensor
        Lower bounds of the test region U.
    U_max : Tensor
        Upper bounds of the test region U.
    N_samples : int
        Number of random test points in U.
    k_max : int
        Max iterations for δ_σ computation.
    manifold_points : Tensor or None
        Known on-manifold points. If None, threshold is computed
        from random samples.

    Returns
    -------
    float
        Δ_σ in [0, 1]. Near 1 indicates excellent manifold quality.
    """
    M = len(U_min)

    # Compute δ_σ for on-manifold points
    if manifold_points is not None and len(manifold_points) > 0:
        on_deltas = []
        for i in range(len(manifold_points)):
            d = delta_sigma(manifold_points[i], model, saddle, sigma, k_max)
            on_deltas.append(d)
        threshold = max(on_deltas) + 1e-6
        median_on = float(sorted(on_deltas)[len(on_deltas) // 2])
    else:
        threshold = 0.5
        median_on = 0.0

    # Compute δ_σ for random points in U
    count_exceeding = 0
    for _ in range(N_samples):
        x = U_min + torch.rand(M) * (U_max - U_min)
        d = delta_sigma(x, model, saddle, sigma, k_max)
        if d > threshold:
            count_exceeding += 1

    indicator_mean = count_exceeding / N_samples
    result = indicator_mean - median_on
    return max(0.0, min(result, 1.0))
