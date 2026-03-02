"""TDD tests for homoclinic/heteroclinic intersection detection.

Tests written FIRST per TDD methodology.
Module: dynamic.analysis.homoclinic

Reference: Appendix I.2, Algorithm 4.
"""

import torch

from dynamic.analysis.manifolds import construct_manifold
from dynamic.analysis.scyfi import FixedPoint
from dynamic.models.plrnn import PLRNN


def _make_model_with_saddle():
    """Create a 2D PLRNN with a known saddle FP."""
    import numpy as np

    torch.manual_seed(0)
    model = PLRNN(M=2)
    with torch.no_grad():
        model.A.copy_(torch.tensor([0.5, 0.5]))
        model.W.copy_(torch.tensor([[0.3, 0.8], [0.0, 0.9]]))
        model.h.copy_(torch.tensor([0.1, -0.1]))

    J = torch.diag(model.A) + model.W
    z_star = torch.linalg.solve(torch.eye(2) - J, model.h)
    evals, evecs = np.linalg.eig(J.detach().numpy())
    fp = FixedPoint(
        z=z_star,
        eigenvalues=evals,
        eigenvectors=evecs,
        classification="saddle",
        region_id=(1, 1),
    )
    return model, fp


class TestHomoclinicDetection:
    """Tests for intersection detection between stable and unstable manifolds."""

    def test_find_intersections_returns_list(self):
        """find_homoclinic_intersections returns a list of tensors."""
        from dynamic.analysis.homoclinic import find_homoclinic_intersections

        model, fp = _make_model_with_saddle()
        stable = construct_manifold(model, fp, sigma=+1, N_s=50, N_iter=5)
        unstable = construct_manifold(model, fp, sigma=-1, N_s=50, N_iter=5)
        result = find_homoclinic_intersections(stable, unstable)
        assert isinstance(result, list)

    def test_no_intersection_self(self):
        """Same manifold passed twice — should still return a list."""
        from dynamic.analysis.homoclinic import find_homoclinic_intersections

        model, fp = _make_model_with_saddle()
        stable = construct_manifold(model, fp, sigma=+1, N_s=30, N_iter=3)
        result = find_homoclinic_intersections(stable, stable)
        assert isinstance(result, list)

    def test_homoclinic_point_finite(self):
        """Any detected intersection points should be finite."""
        from dynamic.analysis.homoclinic import find_homoclinic_intersections

        model, fp = _make_model_with_saddle()
        stable = construct_manifold(model, fp, sigma=+1, N_s=50, N_iter=5)
        unstable = construct_manifold(model, fp, sigma=-1, N_s=50, N_iter=5)
        points = find_homoclinic_intersections(stable, unstable)
        for pt in points:
            assert torch.all(torch.isfinite(pt))
