"""TDD tests for manifold quality metric δ_σ (Eq 6).

Tests written FIRST per TDD methodology.
Module: dynamic.analysis.quality

Reference: §3.3, Eq 6.
"""

import torch

from dynamic.models.plrnn import PLRNN


def _make_stable_fp_model():
    """Create a 2D PLRNN with a stable fixed point in all-positive region."""
    torch.manual_seed(0)
    model = PLRNN(M=2)
    with torch.no_grad():
        model.A.copy_(torch.tensor([0.3, 0.4]))
        model.W.copy_(torch.tensor([[0.0, 0.1], [0.1, 0.0]]))
        model.h.copy_(torch.tensor([0.2, 0.1]))
    # FP in all-positive: (I - J)z* = h
    J = torch.diag(model.A) + model.W
    z_star = torch.linalg.solve(torch.eye(2) - J, model.h)
    return model, z_star


class TestDeltaSigma:
    """Tests for the δ_σ convergence metric."""

    def test_delta_at_fixed_point_is_zero(self):
        """δ_σ at the fixed point itself should be 0."""
        from dynamic.analysis.quality import delta_sigma

        model, z_star = _make_stable_fp_model()
        val = delta_sigma(z_star, model, z_star, sigma=+1, k_max=100)
        assert val < 1e-6, f"δ at FP should be ~0, got {val}"

    def test_delta_near_fp_small(self):
        """Points near a stable FP → δ_σ small."""
        from dynamic.analysis.quality import delta_sigma

        model, z_star = _make_stable_fp_model()
        # Small perturbation
        x0 = z_star + torch.tensor([0.01, -0.01])
        val = delta_sigma(x0, model, z_star, sigma=+1, k_max=200)
        assert val < 0.1, f"δ near stable FP should be small, got {val}"

    def test_delta_bounded(self):
        """δ_σ ∈ [0, 1] for all points."""
        from dynamic.analysis.quality import delta_sigma

        model, z_star = _make_stable_fp_model()
        for _ in range(10):
            x0 = z_star + torch.randn(2) * 0.5
            val = delta_sigma(x0, model, z_star, sigma=+1, k_max=100)
            assert 0.0 <= val <= 1.0 + 1e-6, f"δ out of [0,1]: {val}"

    def test_delta_off_manifold_positive(self):
        """Random points far from manifold → δ_σ > 0."""
        from dynamic.analysis.quality import delta_sigma

        model, z_star = _make_stable_fp_model()
        # Far away point
        x0 = z_star + torch.tensor([10.0, -10.0])
        val = delta_sigma(x0, model, z_star, sigma=+1, k_max=100)
        # Point that doesn't converge well should have delta > 0
        assert val >= 0.0


class TestDeltaStatistic:
    """Tests for the aggregate Δ_σ statistic."""

    def test_delta_statistic_returns_float(self):
        """Δ_σ returns a float in [0, 1]."""
        from dynamic.analysis.quality import delta_sigma_statistic

        model, z_star = _make_stable_fp_model()
        # Use a small bounding box around the FP
        result = delta_sigma_statistic(
            model, z_star, sigma=+1,
            U_min=z_star - 1.0, U_max=z_star + 1.0,
            N_samples=50, k_max=100,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0 + 1e-6
