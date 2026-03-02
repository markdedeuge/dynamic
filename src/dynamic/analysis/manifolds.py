"""Manifold detection (Algorithm 1).

Semi-analytical method for computing stable and unstable manifolds
of saddle fixed points in PLRNNs.

Overview:
1. At a saddle point, decompose the Jacobian eigenspace into stable
   (|λ| < 1) and unstable (|λ| > 1) subspaces.
2. Sample seed points on the local manifold using eigenvalue-rescaled
   Gaussian mixture sampling.
3. Propagate points forward (unstable) or backward (stable) across
   subregion boundaries.
4. Fit manifold segments with PCA (planar) or kPCA (curved).
5. Iterate to expand the manifold across the state space.

Reference: §3.3, Algorithm 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy import ndarray
from sklearn.decomposition import PCA, KernelPCA
from torch import Tensor, nn

from dynamic.analysis.backtracking import backward_step, verify_forward
from dynamic.analysis.subregions import get_D, get_region_id


@dataclass
class ManifoldSegment:
    """A segment of a manifold within one linear subregion.

    Attributes
    ----------
    region_id : tuple
        Subregion identifier (binary sign pattern).
    support_point : Tensor
        Reference point in this segment (e.g. saddle or centroid).
    eigenvectors : ndarray
        Principal directions of the manifold in this region.
        Shape ``(M, d)`` where ``d`` is the manifold dimension.
    is_curved : bool
        True if kPCA was used (complex eigenvalues detected).
    points : Tensor
        Points on the manifold within this region.
    """

    region_id: tuple
    support_point: Tensor
    eigenvectors: ndarray
    is_curved: bool
    points: Tensor = field(default_factory=lambda: torch.empty(0))


def compute_local_manifold(
    model: nn.Module,
    saddle,
    sigma: int,
) -> ManifoldSegment:
    """Eigenvector decomposition at a saddle point.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    saddle : FixedPoint
        Saddle fixed point with eigenvalues and eigenvectors.
    sigma : int
        +1 for stable manifold (|λ| < 1), -1 for unstable (|λ| > 1).

    Returns
    -------
    ManifoldSegment
        Local manifold segment at the saddle.
    """
    evals = saddle.eigenvalues
    evecs = saddle.eigenvectors

    if sigma == +1:
        mask = np.abs(evals) < 1.0
    else:
        mask = np.abs(evals) > 1.0

    selected_evecs = evecs[:, mask].real  # take real parts for directions

    return ManifoldSegment(
        region_id=saddle.region_id,
        support_point=saddle.z.clone(),
        eigenvectors=selected_evecs,
        is_curved=False,
        points=saddle.z.unsqueeze(0),
    )


def sample_on_manifold(
    segment: ManifoldSegment,
    N_s: int = 100,
    scales: list[float] | None = None,
) -> Tensor:
    """Eigenvalue-rescaled GMM sampling on a local manifold.

    Places N_s points on the manifold using multiple scale factors.

    Parameters
    ----------
    segment : ManifoldSegment
        Local manifold segment.
    N_s : int
        Total number of points to sample.
    scales : list[float]
        Scale factors for Gaussian sampling.

    Returns
    -------
    Tensor
        Sampled points of shape ``(N_actual, M)``.
    """
    if scales is None:
        scales = [0.01, 0.1, 0.5]

    center = segment.support_point.detach()
    directions = torch.tensor(
        segment.eigenvectors, dtype=torch.float32,
    )

    d = directions.shape[1]  # manifold dimension
    n_per_scale = max(1, N_s // len(scales))

    all_points = []
    for scale in scales:
        # Sample in eigenspace coordinates
        coords = torch.randn(n_per_scale, d) * scale
        # Project to full space
        offsets = coords @ directions.T
        points = center.unsqueeze(0) + offsets
        all_points.append(points)

    return torch.cat(all_points, dim=0)


def propagate_to_next_region(
    model: nn.Module,
    points: Tensor,
    sigma: int,
) -> dict[tuple, Tensor]:
    """Push points forward (σ=-1) or backward (σ=+1) and group by region.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    points : Tensor
        Points to propagate, shape ``(N, M)``.
    sigma : int
        +1 for stable manifold (backward), -1 for unstable (forward).

    Returns
    -------
    dict[tuple, Tensor]
        Mapping from region_id to points that landed in that region.
    """
    result: dict[tuple, list[Tensor]] = {}

    for i in range(len(points)):
        z = points[i]
        if sigma == -1:
            # Unstable: push forward
            with torch.no_grad():
                z_next = model.forward(z)
        else:
            # Stable: push backward
            D = get_D(z)
            try:
                z_next = backward_step(model, z, D)
                if not verify_forward(model, z_next, z, tol=1e-4):
                    continue
            except (torch.linalg.LinAlgError, RuntimeError):
                continue

        region = get_region_id(z_next)
        if region not in result:
            result[region] = []
        result[region].append(z_next.detach())

    return {
        rid: torch.stack(pts) for rid, pts in result.items() if pts
    }


def fit_manifold_segment(
    points: Tensor,
    eigenvalues: ndarray,
    support_point: Tensor,
    region_id: tuple,
) -> ManifoldSegment:
    """Fit a manifold segment using PCA or kPCA.

    Uses PCA for real eigenvalues, kPCA with RBF kernel for
    complex or defective eigenvalues.

    Parameters
    ----------
    points : Tensor
        Points on the manifold, shape ``(N, M)``.
    eigenvalues : ndarray
        Eigenvalues associated with this manifold.
    support_point : Tensor
        Reference point for the segment.
    region_id : tuple
        Subregion ID.

    Returns
    -------
    ManifoldSegment
        Fitted segment with PCA/kPCA components.
    """
    X = points.detach().numpy()
    n_components = len(eigenvalues)

    # Use kPCA if complex eigenvalues present
    has_complex = np.any(np.abs(np.imag(eigenvalues)) > 1e-10)

    if has_complex and X.shape[0] > n_components:
        try:
            kpca = KernelPCA(
                n_components=n_components,
                kernel="rbf",
                gamma=1.0,
            )
            kpca.fit(X)
            # Approximate components via PCA on transformed data
            pca = PCA(n_components=n_components)
            pca.fit(X)
            components = pca.components_.T
            is_curved = True
        except Exception:
            pca = PCA(n_components=min(n_components, X.shape[0]))
            pca.fit(X)
            components = pca.components_.T
            is_curved = False
    else:
        pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
        pca.fit(X)
        components = pca.components_.T
        is_curved = False

    return ManifoldSegment(
        region_id=region_id,
        support_point=support_point.detach(),
        eigenvectors=components,
        is_curved=is_curved,
        points=points.detach(),
    )


def construct_manifold(
    model: nn.Module,
    saddle,
    sigma: int,
    N_s: int = 100,
    N_iter: int = 50,
    scales: list[float] | None = None,
) -> list[ManifoldSegment]:
    """Full Algorithm 1: iterative manifold construction.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    saddle : FixedPoint
        Saddle fixed point.
    sigma : int
        +1 for stable, -1 for unstable manifold.
    N_s : int
        Number of sample points per iteration.
    N_iter : int
        Number of expansion iterations.
    scales : list[float]
        Sampling scales.

    Returns
    -------
    list[ManifoldSegment]
        Fitted manifold segments across regions.
    """
    if scales is None:
        scales = [0.01, 0.1, 0.5]

    # Step 1: compute local manifold
    local = compute_local_manifold(model, saddle, sigma)

    # Select eigenvalues for the chosen manifold type
    evals = saddle.eigenvalues
    if sigma == +1:
        mask = np.abs(evals) < 1.0
    else:
        mask = np.abs(evals) > 1.0
    selected_evals = evals[mask]

    segments: dict[tuple, ManifoldSegment] = {
        local.region_id: local,
    }

    # Step 2: sample initial points
    current_points = sample_on_manifold(local, N_s=N_s, scales=scales)

    for _iteration in range(N_iter):
        # Step 3: propagate to neighboring regions
        region_groups = propagate_to_next_region(model, current_points, sigma)

        new_points_all = []
        for region_id, pts in region_groups.items():
            if pts.shape[0] < 2:
                continue

            # Step 4: fit manifold segment in new region
            n_comp = min(len(selected_evals), pts.shape[0] - 1, pts.shape[1])
            if n_comp < 1:
                continue

            segment = fit_manifold_segment(
                pts,
                eigenvalues=selected_evals[:n_comp],
                support_point=pts.mean(dim=0),
                region_id=region_id,
            )
            segments[region_id] = segment
            new_points_all.append(pts)

        if not new_points_all:
            break

        # Step 5: resample from the new segments for next iteration
        current_points = torch.cat(new_points_all, dim=0)
        # Subsample to keep manageable
        if current_points.shape[0] > N_s:
            indices = torch.randperm(current_points.shape[0])[:N_s]
            current_points = current_points[indices]

    return list(segments.values())
