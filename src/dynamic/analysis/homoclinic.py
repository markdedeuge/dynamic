"""Homoclinic and heteroclinic intersection detection.

Finds intersections between stable and unstable manifolds.
In 2D, uses geometric segment intersection. For higher dimensions,
uses nearest-neighbor proximity detection.

Reference: Appendix I.2, Algorithm 4.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dynamic.analysis.manifolds import ManifoldSegment


def _segments_intersect_2d(
    p1: Tensor,
    p2: Tensor,
    p3: Tensor,
    p4: Tensor,
) -> Tensor | None:
    """Check if line segments (p1-p2) and (p3-p4) intersect in 2D.

    Returns the intersection point or None.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross.item()) < 1e-12:
        return None  # parallel

    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross

    if 0 <= t.item() <= 1 and 0 <= u.item() <= 1:
        return p1 + t * d1
    return None


def find_homoclinic_intersections(
    stable_manifold: list[ManifoldSegment],
    unstable_manifold: list[ManifoldSegment],
    proximity_threshold: float = 0.05,
) -> list[Tensor]:
    """Find intersection points between stable and unstable manifolds.

    For 2D models, uses geometric line segment intersection.
    For higher dimensions, uses nearest-neighbor proximity.

    Parameters
    ----------
    stable_manifold : list[ManifoldSegment]
        Segments of the stable manifold.
    unstable_manifold : list[ManifoldSegment]
        Segments of the unstable manifold.
    proximity_threshold : float
        Maximum distance to consider as an intersection (for ≥ 3D).

    Returns
    -------
    list[Tensor]
        List of intersection points.
    """
    intersections: list[Tensor] = []

    for s_seg in stable_manifold:
        if s_seg.points.numel() == 0:
            continue
        s_pts = s_seg.points

        for u_seg in unstable_manifold:
            if u_seg.points.numel() == 0:
                continue
            u_pts = u_seg.points

            M = s_pts.shape[1] if s_pts.dim() > 1 else s_pts.shape[0]

            if M == 2 and s_pts.dim() > 1 and u_pts.dim() > 1:
                # 2D: geometric segment intersection
                for i in range(len(s_pts) - 1):
                    for j in range(len(u_pts) - 1):
                        pt = _segments_intersect_2d(
                            s_pts[i],
                            s_pts[i + 1],
                            u_pts[j],
                            u_pts[j + 1],
                        )
                        if pt is not None:
                            # Check for duplicates
                            is_dup = False
                            for existing in intersections:
                                if torch.allclose(pt, existing, atol=1e-4):
                                    is_dup = True
                                    break
                            if not is_dup:
                                intersections.append(pt.detach())
            else:
                # Higher-D: proximity detection
                if s_pts.dim() < 2 or u_pts.dim() < 2:
                    continue
                # Compute pairwise distances
                dists = torch.cdist(s_pts, u_pts)
                close = (dists < proximity_threshold).nonzero()
                for idx in close:
                    midpoint = (s_pts[idx[0]] + u_pts[idx[1]]) / 2
                    is_dup = False
                    for existing in intersections:
                        if torch.allclose(midpoint, existing, atol=1e-4):
                            is_dup = True
                            break
                    if not is_dup:
                        intersections.append(midpoint.detach())

    return intersections


def analytical_homoclinic_2d(
    model,
    saddle,
    N_s: int = 500,
    N_iter: int = 30,
) -> list[Tensor]:
    """Algorithm 4 (Appx I.2): analytical homoclinic detection for 2D.

    For 2D PL maps, generates dense trajectory-based point clouds on
    both stable and unstable manifolds, then checks for geometric
    intersections.

    Parameters
    ----------
    model : nn.Module
        2D PLRNN or PLMapModel with A, W, h parameters.
    saddle : FixedPoint
        Saddle fixed point with eigenvalues and eigenvectors.
    N_s : int
        Number of sample points / initial conditions.
    N_iter : int
        Number of forward/backward iterations per trajectory.

    Returns
    -------
    list[Tensor]
        Detected homoclinic intersection points.
    """
    import numpy as np

    evals = saddle.eigenvalues
    evecs = saddle.eigenvectors

    # Separate stable (|λ|<1) and unstable (|λ|>1) eigenvectors
    stable_mask = np.abs(evals) < 1.0
    unstable_mask = np.abs(evals) > 1.0

    stable_dirs = torch.tensor(
        evecs[:, stable_mask].real,
        dtype=torch.float32,
    )
    unstable_dirs = torch.tensor(
        evecs[:, unstable_mask].real,
        dtype=torch.float32,
    )

    p = saddle.z.detach()

    # Generate dense unstable manifold by forward-iterating from
    # perturbations along the unstable eigenvector(s)
    unstable_trajectories: list[Tensor] = []
    for sign in [-1.0, 1.0]:
        for scale in [0.001, 0.005, 0.01, 0.05, 0.1]:
            for col in range(unstable_dirs.shape[1]):
                z = p + sign * scale * unstable_dirs[:, col]
                traj = [z.detach().clone()]
                with torch.no_grad():
                    for _ in range(N_iter):
                        z = model.forward(z)
                        if not torch.all(torch.isfinite(z)):
                            break
                        traj.append(z.clone())
                if len(traj) > 1:
                    unstable_trajectories.append(torch.stack(traj))

    # Generate dense stable manifold by forward-iterating many ICs
    # and keeping trajectories that converge to the saddle
    stable_trajectories: list[Tensor] = []

    # Method 1: perturb along stable eigenvector and forward-iterate
    for sign in [-1.0, 1.0]:
        for scale in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
            for col in range(stable_dirs.shape[1]):
                z = p + sign * scale * stable_dirs[:, col]
                traj = [z.detach().clone()]
                with torch.no_grad():
                    for _ in range(N_iter):
                        z = model.forward(z)
                        if not torch.all(torch.isfinite(z)):
                            break
                        traj.append(z.clone())
                        # If converged to saddle, this is on stable manifold
                        if torch.norm(z - p) < 1e-6:
                            break
                if len(traj) > 1:
                    stable_trajectories.append(torch.stack(traj))

    # Method 2: random ICs — keep those whose trajectories pass near saddle
    n_random = max(N_s, 500)
    for _ in range(n_random):
        z = p + torch.randn(2) * 3.0
        traj = [z.detach().clone()]
        min_dist = float("inf")
        with torch.no_grad():
            for _ in range(N_iter):
                z = model.forward(z)
                if not torch.all(torch.isfinite(z)):
                    break
                d = torch.norm(z - p).item()
                if d < min_dist:
                    min_dist = d
                traj.append(z.clone())
        # Keep trajectories that come close to saddle
        if min_dist < 0.05 and len(traj) > 3:
            stable_trajectories.append(torch.stack(traj))

    # Collect all trajectory points into flat tensors for fast proximity check
    u_all = (
        torch.cat(unstable_trajectories) if unstable_trajectories else torch.empty(0, 2)
    )
    s_all = torch.cat(stable_trajectories) if stable_trajectories else torch.empty(0, 2)

    if u_all.shape[0] == 0 or s_all.shape[0] == 0:
        return []

    # Remove points too close to the saddle to avoid trivial intersections
    u_dists = torch.norm(u_all - p, dim=1)
    s_dists = torch.norm(s_all - p, dim=1)
    u_far = u_all[u_dists > 0.01]
    s_far = s_all[s_dists > 0.01]

    if u_far.shape[0] == 0 or s_far.shape[0] == 0:
        return []

    # Vectorized proximity detection
    dists = torch.cdist(u_far, s_far)
    close = (dists < 0.05).nonzero()

    intersections: list[Tensor] = []
    for idx in close:
        midpoint = (u_far[idx[0]] + s_far[idx[1]]) / 2
        is_dup = any(torch.allclose(midpoint, ex, atol=0.05) for ex in intersections)
        if not is_dup:
            intersections.append(midpoint.detach())

    return intersections


def _find_boundary_crossings(points: Tensor, dim: int) -> list[Tensor]:
    """Find points where a sequence crosses z_dim = 0.

    Returns interpolated crossing points.
    """
    crossings: list[Tensor] = []
    for i in range(len(points) - 1):
        v0 = points[i, dim].item()
        v1 = points[i + 1, dim].item()
        if v0 * v1 < 0:  # sign change
            t = abs(v0) / (abs(v0) + abs(v1))
            crossing = points[i] * (1 - t) + points[i + 1] * t
            crossings.append(crossing.detach())
    return crossings
