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
