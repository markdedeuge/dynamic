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
    p1: Tensor, p2: Tensor, p3: Tensor, p4: Tensor,
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
                            s_pts[i], s_pts[i + 1],
                            u_pts[j], u_pts[j + 1],
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

    For 2D PL maps, computes stable and unstable manifolds and checks
    for intersections at subregion boundaries (z_i = 0 hyperplanes).

    A homoclinic point exists when both manifolds cross the same
    boundary segment — this is a fold point where the manifold is
    tangent to the boundary.

    Parameters
    ----------
    model : nn.Module
        2D PLRNN or PLMapModel with A, W, h parameters.
    saddle : FixedPoint
        Saddle fixed point with eigenvalues and eigenvectors.
    N_s : int
        Number of sample points for manifold construction.
    N_iter : int
        Number of manifold expansion iterations.

    Returns
    -------
    list[Tensor]
        Detected homoclinic intersection points.
    """
    from dynamic.analysis.manifolds import construct_manifold

    # Compute both manifolds with enough resolution
    stable = construct_manifold(
        model, saddle, sigma=+1, N_s=N_s, N_iter=N_iter,
    )
    unstable = construct_manifold(
        model, saddle, sigma=-1, N_s=N_s, N_iter=N_iter,
    )

    # Find intersections using geometric method
    intersections = find_homoclinic_intersections(
        stable, unstable, proximity_threshold=0.1,
    )

    # Additionally check for boundary crossings (fold points)
    # In 2D, a fold occurs when the manifold crosses z_i = 0
    for s_seg in stable:
        if s_seg.points.numel() == 0 or s_seg.points.dim() < 2:
            continue
        s_pts = s_seg.points

        for u_seg in unstable:
            if u_seg.points.numel() == 0 or u_seg.points.dim() < 2:
                continue
            u_pts = u_seg.points

            # Check z_1 = 0 boundary crossings
            for dim in range(2):
                s_crossings = _find_boundary_crossings(s_pts, dim)
                u_crossings = _find_boundary_crossings(u_pts, dim)

                # Match crossing points
                for sc in s_crossings:
                    for uc in u_crossings:
                        if torch.norm(sc - uc) < 0.1:
                            is_dup = any(
                                torch.allclose(sc, ex, atol=1e-3)
                                for ex in intersections
                            )
                            if not is_dup:
                                mid = (sc + uc) / 2
                                intersections.append(mid.detach())

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
