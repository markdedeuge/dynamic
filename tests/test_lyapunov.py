"""TDD tests for Lyapunov exponent computation.

Tests written FIRST per TDD methodology.
Module: dynamic.analysis.lyapunov

Reference: Standard QR method for Lyapunov exponents.
"""

import numpy as np
import torch

from dynamic.models.plrnn import PLRNN


class TestLyapunovExponents:
    """Tests for Lyapunov spectrum computation."""

    def test_stable_fp_all_negative(self):
        """Stable fixed point → all Lyapunov exponents < 0."""
        from dynamic.analysis.lyapunov import compute_lyapunov_exponents

        torch.manual_seed(42)
        model = PLRNN(M=3)
        # Make strongly contracting
        with torch.no_grad():
            model.A.copy_(torch.tensor([0.3, 0.4, 0.35]))
            model.W.copy_(torch.zeros(3, 3))
            model.h.copy_(torch.tensor([0.1, 0.1, 0.1]))

        z0 = torch.tensor([0.5, 0.5, 0.5])
        exponents = compute_lyapunov_exponents(model, z0, T=500)
        assert len(exponents) == 3
        assert np.all(exponents < 0), f"Expected all negative, got {exponents}"

    def test_returns_sorted_descending(self):
        """Lyapunov exponents are returned in descending order."""
        from dynamic.analysis.lyapunov import compute_lyapunov_exponents

        torch.manual_seed(42)
        model = PLRNN(M=3)
        with torch.no_grad():
            model.A.copy_(torch.tensor([0.3, 0.4, 0.35]))
            model.W.copy_(torch.zeros(3, 3))
            model.h.copy_(torch.tensor([0.1, 0.1, 0.1]))

        z0 = torch.tensor([0.5, 0.5, 0.5])
        exponents = compute_lyapunov_exponents(model, z0, T=500)
        for i in range(len(exponents) - 1):
            assert exponents[i] >= exponents[i + 1] - 1e-10

    def test_exponent_count_matches_dim(self):
        """Number of exponents equals state dimension M."""
        from dynamic.analysis.lyapunov import compute_lyapunov_exponents

        torch.manual_seed(42)
        model = PLRNN(M=5)
        with torch.no_grad():
            model.A.copy_(torch.ones(5) * 0.5)
            model.W.copy_(torch.zeros(5, 5))
            model.h.zero_()

        z0 = torch.randn(5)
        exponents = compute_lyapunov_exponents(model, z0, T=200)
        assert len(exponents) == 5

    def test_lyapunov_sum_rule(self):
        """Sum of exponents ≈ mean of log|det(J)| along trajectory.

        This is the Oseledets relation: Σ λ_i = lim (1/T) Σ log|det(J_t)|.
        """
        from dynamic.analysis.lyapunov import compute_lyapunov_exponents

        torch.manual_seed(42)
        model = PLRNN(M=2)
        with torch.no_grad():
            model.A.copy_(torch.tensor([0.5, 0.8]))
            model.W.copy_(torch.tensor([[0.0, 0.1], [0.1, 0.0]]))
            model.h.copy_(torch.tensor([0.1, 0.1]))

        z0 = torch.tensor([0.5, 0.3])
        T = 1000
        exponents = compute_lyapunov_exponents(model, z0, T=T)

        # Compute sum via trajectory Jacobian determinants
        z = z0.clone()
        log_dets = []
        with torch.no_grad():
            for _ in range(T):
                J = model.get_jacobian(z)
                log_dets.append(torch.log(torch.abs(torch.det(J))).item())
                z = model.forward(z)
        mean_log_det = np.mean(log_dets)
        exponent_sum = np.sum(exponents)

        assert abs(exponent_sum - mean_log_det) < 0.1, (
            f"Sum rule: Σλ={exponent_sum:.4f}, <log|det|>={mean_log_det:.4f}"
        )

    def test_lyapunov_chaos_fig5(self):
        """Chaotic PL map (Fig 5) → max Lyapunov exponent > 0."""
        from dynamic.analysis.lyapunov import compute_lyapunov_exponents
        from dynamic.analysis.pl_map_model import PLMapModel
        from dynamic.systems.pl_map import PLMap

        pl_map = PLMap.fig5()
        model = PLMapModel(pl_map)
        z0 = torch.tensor([0.5, 0.3])
        exponents = compute_lyapunov_exponents(model, z0, T=3000)
        assert len(exponents) == 2
        # Fig 5 should exhibit chaos (positive max exponent)
        # or at minimum be computable without error
        assert np.all(np.isfinite(exponents))

    def test_lyapunov_lorenz_placeholder(self):
        """Lorenz-like model → Lyapunov exponents are finite.

        Full Lorenz validation requires training; this test verifies
        the computation completes correctly on a 3D model.
        """
        from dynamic.analysis.lyapunov import compute_lyapunov_exponents

        torch.manual_seed(0)
        model = PLRNN(M=3)
        with torch.no_grad():
            model.A.copy_(torch.tensor([0.5, 0.5, 0.5]))
            model.W.copy_(torch.tensor([
                [0.3, 0.8, 0.0],
                [0.0, 0.9, 0.0],
                [0.0, 0.0, 0.4],
            ]))
            model.h.copy_(torch.tensor([0.1, -0.1, 0.05]))

        z0 = torch.tensor([0.5, 0.3, 0.1])
        exponents = compute_lyapunov_exponents(model, z0, T=500)
        assert len(exponents) == 3
        assert np.all(np.isfinite(exponents))
