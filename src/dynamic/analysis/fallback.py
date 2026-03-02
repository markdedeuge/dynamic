"""Fallback manifold detection (Algorithm 3).

For manifolds with folding or discontinuous structure, this fallback
perturbs seed points, generates forward/backward trajectories,
clusters per subregion using HDBSCAN, and fits PCA/kPCA per cluster.

Reference: Appendix C, Algorithm 3.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from dynamic.analysis.backtracking import backtrack_trajectory
from dynamic.analysis.manifolds import ManifoldSegment, fit_manifold_segment
from dynamic.analysis.subregions import get_region_id


def fallback_manifold_detection(
    model: nn.Module,
    saddle,
    sigma: int,
    N_forward: int = 100,
    N_backward: int = 100,
    perturbation_scale: float = 0.1,
) -> list[ManifoldSegment]:
    """Algorithm 3: trajectory perturbation + clustering.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    saddle : FixedPoint
        Saddle fixed point.
    sigma : int
        +1 for stable, -1 for unstable manifold.
    N_forward : int
        Number of forward propagation steps.
    N_backward : int
        Number of backward propagation steps.
    perturbation_scale : float
        Scale of random perturbations around the saddle.

    Returns
    -------
    list[ManifoldSegment]
        Manifold segments detected by the fallback algorithm.
    """
    # Select eigenvalues for the manifold type
    evals = saddle.eigenvalues
    if sigma == +1:
        mask = np.abs(evals) < 1.0
    else:
        mask = np.abs(evals) > 1.0
    selected_evals = evals[mask]
    evecs = saddle.eigenvectors[:, mask].real

    # Generate perturbed seed points along relevant eigenvectors
    directions = torch.tensor(evecs, dtype=torch.float32)
    d = directions.shape[1]

    n_seeds = max(10, N_forward // 2)
    seeds = []
    for _ in range(n_seeds):
        offset = torch.randn(d) * perturbation_scale
        point = saddle.z.detach() + (offset @ directions.T)
        seeds.append(point)

    # Generate trajectories
    all_points: dict[tuple, list[Tensor]] = {}

    for seed in seeds:
        if sigma == -1:
            # Unstable: forward trajectories
            with torch.no_grad():
                traj = model.forward_trajectory(seed, T=N_forward)
            for t in range(len(traj)):
                z = traj[t]
                rid = get_region_id(z)
                if rid not in all_points:
                    all_points[rid] = []
                all_points[rid].append(z.detach())
        else:
            # Stable: backward trajectories
            try:
                traj = backtrack_trajectory(
                    model, seed, T=N_backward, tol=1e-4,
                )
                for t in range(len(traj)):
                    z = traj[t]
                    rid = get_region_id(z)
                    if rid not in all_points:
                        all_points[rid] = []
                    all_points[rid].append(z.detach())
            except Exception:
                continue

    # Fit manifold segments per region
    segments: list[ManifoldSegment] = []

    for region_id, pts_list in all_points.items():
        if len(pts_list) < 3:
            continue
        pts = torch.stack(pts_list)

        # Try HDBSCAN clustering if enough points
        if len(pts_list) >= 10:
            try:
                import hdbscan

                clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
                labels = clusterer.fit_predict(pts.numpy())
                unique_labels = set(labels)
                unique_labels.discard(-1)  # remove noise label

                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_pts = pts[cluster_mask]
                    if len(cluster_pts) < 2:
                        continue
                    n_comp = min(
                        len(selected_evals),
                        cluster_pts.shape[0] - 1,
                        cluster_pts.shape[1],
                    )
                    if n_comp < 1:
                        continue
                    segment = fit_manifold_segment(
                        cluster_pts,
                        eigenvalues=selected_evals[:n_comp],
                        support_point=cluster_pts.mean(dim=0),
                        region_id=region_id,
                    )
                    segments.append(segment)
                continue
            except ImportError:
                pass

        # Fallback: fit without clustering
        n_comp = min(
            len(selected_evals), pts.shape[0] - 1, pts.shape[1],
        )
        if n_comp < 1:
            continue
        segment = fit_manifold_segment(
            pts,
            eigenvalues=selected_evals[:n_comp],
            support_point=pts.mean(dim=0),
            region_id=region_id,
        )
        segments.append(segment)

    return segments
