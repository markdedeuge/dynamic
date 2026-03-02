"""TDD tests for manifold detection (Algorithm 1 and 3).

Tests written FIRST per TDD methodology.
Module: dynamic.analysis.manifolds, dynamic.analysis.fallback

Reference: §3.3, Algorithm 1, Appendix C.
"""

import numpy as np
import torch

from dynamic.analysis.scyfi import FixedPoint
from dynamic.models.plrnn import PLRNN


def _make_saddle_model_and_fp():
    """Create a 2D PLRNN with a known saddle fixed point.

    Returns (model, FixedPoint) where the FP is a saddle
    with one stable and one unstable eigenvalue.
    """
    torch.manual_seed(0)
    model = PLRNN(M=2)
    # Set params so the all-positive region has a saddle FP.
    # J = diag(A) + W @ I = [[A0+W00, W01], [W10, A1+W11]]
    # Want eigenvalues with |λ1| < 1 < |λ2|.
    with torch.no_grad():
        model.A.copy_(torch.tensor([0.5, 0.5]))
        model.W.copy_(torch.tensor([[0.3, 0.8], [0.0, 0.9]]))
        model.h.copy_(torch.tensor([0.1, -0.1]))

    # Solve for FP in the all-positive region: (I - J) z* = h
    J = torch.diag(model.A) + model.W  # D = I in all-positive
    z_star = torch.linalg.solve(torch.eye(2) - J, model.h)
    assert z_star[0] > 0 and z_star[1] > 0, "FP must be in all-positive region"

    evals, evecs = np.linalg.eig(J.detach().numpy())
    return model, FixedPoint(
        z=z_star,
        eigenvalues=evals,
        eigenvectors=evecs,
        classification="saddle",
        region_id=(1, 1),
    )


class TestLocalManifold:
    """Tests for local manifold computation at a saddle point."""

    def test_local_manifold_eigenvectors(self):
        """Local manifold at saddle is spanned by eigenvectors."""
        from dynamic.analysis.manifolds import compute_local_manifold

        model, fp = _make_saddle_model_and_fp()
        # sigma=+1 → stable manifold → eigenvectors with |λ| < 1
        segment = compute_local_manifold(model, fp, sigma=+1)
        assert segment is not None
        assert segment.support_point is not None
        assert torch.allclose(segment.support_point, fp.z, atol=1e-6)

    def test_local_manifold_dimension(self):
        """Manifold dimension = number of relevant eigenvalues."""
        from dynamic.analysis.manifolds import compute_local_manifold

        model, fp = _make_saddle_model_and_fp()
        # Count stable eigenvalues (|λ| < 1)
        n_stable = np.sum(np.abs(fp.eigenvalues) < 1)
        segment = compute_local_manifold(model, fp, sigma=+1)
        assert segment.eigenvectors.shape[1] == n_stable

    def test_local_manifold_unstable(self):
        """sigma=-1 → unstable manifold → eigenvectors with |λ| > 1."""
        from dynamic.analysis.manifolds import compute_local_manifold

        model, fp = _make_saddle_model_and_fp()
        n_unstable = np.sum(np.abs(fp.eigenvalues) > 1)
        segment = compute_local_manifold(model, fp, sigma=-1)
        assert segment.eigenvectors.shape[1] == n_unstable


class TestSampling:
    """Tests for eigenvalue-rescaled GMM sampling."""

    def test_sample_shape(self):
        """Sampled points have shape (N_s, M)."""
        from dynamic.analysis.manifolds import (
            compute_local_manifold,
            sample_on_manifold,
        )

        model, fp = _make_saddle_model_and_fp()
        segment = compute_local_manifold(model, fp, sigma=-1)
        points = sample_on_manifold(segment, N_s=50, scales=[0.01, 0.1, 0.5])
        assert points.shape[0] <= 50  # may be fewer after rejection
        assert points.shape[1] == 2  # M=2

    def test_sample_near_saddle(self):
        """Sampled points are near the saddle point."""
        from dynamic.analysis.manifolds import (
            compute_local_manifold,
            sample_on_manifold,
        )

        model, fp = _make_saddle_model_and_fp()
        segment = compute_local_manifold(model, fp, sigma=-1)
        points = sample_on_manifold(segment, N_s=100, scales=[0.01])
        dists = torch.norm(points - fp.z, dim=1)
        assert dists.max() < 0.5, f"Points too far from saddle: max={dists.max():.3f}"

    def test_sample_multi_scale(self):
        """Sampled points cover multiple scales."""
        from dynamic.analysis.manifolds import (
            compute_local_manifold,
            sample_on_manifold,
        )

        model, fp = _make_saddle_model_and_fp()
        segment = compute_local_manifold(model, fp, sigma=-1)
        points = sample_on_manifold(segment, N_s=200, scales=[0.01, 0.1, 0.5])
        dists = torch.norm(points - fp.z, dim=1)
        # Should have both close and far points
        assert dists.min() < 0.05
        assert dists.max() > 0.05


class TestManifoldFitting:
    """Tests for PCA/kPCA manifold fitting."""

    def test_pca_recovers_line(self):
        """PCA on collinear 2D points → 1 principal component."""
        from dynamic.analysis.manifolds import fit_manifold_segment

        # Points along a line in 2D
        t = torch.linspace(-1, 1, 50).unsqueeze(1)
        direction = torch.tensor([[1.0, 0.5]])
        points = t * direction + torch.tensor([1.0, 2.0])

        segment = fit_manifold_segment(
            points, eigenvalues=np.array([0.5]),
            support_point=torch.tensor([1.0, 2.0]),
            region_id=(1, 1),
        )
        assert not segment.is_curved
        assert segment.eigenvectors.shape[1] == 1

    def test_pca_2d_plane(self):
        """PCA on 3D points on a plane → 2 components."""
        from dynamic.analysis.manifolds import fit_manifold_segment

        rng = np.random.default_rng(42)
        t = rng.standard_normal((100, 2))
        basis = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        points = torch.tensor(t @ basis + [1, 2, 3], dtype=torch.float32)

        segment = fit_manifold_segment(
            points, eigenvalues=np.array([0.3, 0.7]),
            support_point=torch.tensor([1.0, 2.0, 3.0]),
            region_id=(1, 1, 1),
        )
        assert not segment.is_curved
        assert segment.eigenvectors.shape[1] == 2


class TestFullManifoldConstruction:
    """Tests for full Algorithm 1 construction."""

    def test_manifold_2d_toy(self):
        """Full Algo 1 on 2D PLRNN produces non-empty manifold."""
        from dynamic.analysis.manifolds import construct_manifold

        model, fp = _make_saddle_model_and_fp()
        segments = construct_manifold(
            model, fp, sigma=-1, N_s=50, N_iter=5,
        )
        assert len(segments) >= 1, "Manifold should have at least 1 segment"

    def test_manifold_spans_regions(self):
        """Manifold construction visits ≥ 1 subregion."""
        from dynamic.analysis.manifolds import construct_manifold

        model, fp = _make_saddle_model_and_fp()
        segments = construct_manifold(
            model, fp, sigma=-1, N_s=100, N_iter=10,
        )
        regions = {s.region_id for s in segments}
        assert len(regions) >= 1

    def test_manifold_stable_unstable_differ(self):
        """sigma=+1 (stable) and sigma=-1 (unstable) produce different manifolds."""
        from dynamic.analysis.manifolds import construct_manifold

        model, fp = _make_saddle_model_and_fp()
        stable = construct_manifold(model, fp, sigma=+1, N_s=50, N_iter=5)
        unstable = construct_manifold(model, fp, sigma=-1, N_s=50, N_iter=5)
        # At minimum, the eigenvectors should differ
        s_evecs = stable[0].eigenvectors
        u_evecs = unstable[0].eigenvectors
        assert not np.allclose(s_evecs, u_evecs, atol=1e-3)


class TestFallback:
    """Tests for Algorithm 3 fallback."""

    def test_fallback_produces_segments(self):
        """Fallback algorithm produces at least 1 segment."""
        from dynamic.analysis.fallback import fallback_manifold_detection

        model, fp = _make_saddle_model_and_fp()
        segments = fallback_manifold_detection(
            model, fp, sigma=-1, N_forward=20, N_backward=20,
        )
        assert len(segments) >= 1
