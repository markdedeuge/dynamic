"""TDD tests for map inversion / backtracking (Algorithm 2).

Tests written FIRST per TDD methodology.
Module: dynamic.analysis.backtracking

Reference: Appendix C of the paper.
"""

import pytest
import torch

from dynamic.models.plrnn import PLRNN


@pytest.fixture()
def simple_model():
    """A small PLRNN with known stable behaviour."""
    torch.manual_seed(42)
    model = PLRNN(M=3)
    # Make near-identity so inversions are well-conditioned
    with torch.no_grad():
        model.A.copy_(torch.tensor([0.9, 0.8, 0.85]))
        model.W.copy_(torch.tensor([
            [0.0, 0.05, -0.03],
            [-0.04, 0.0, 0.06],
            [0.02, -0.05, 0.0],
        ]))
        model.h.copy_(torch.tensor([0.1, -0.1, 0.05]))
    return model


class TestBackwardStep:
    """Tests for single backward step."""

    def test_backward_forward_consistency(self, simple_model):
        """F(F⁻¹(z)) == z to < 1e-8."""
        from dynamic.analysis.backtracking import backward_step

        z = torch.tensor([0.5, 0.3, 0.8])
        D = simple_model.get_D(z)
        z_prev = backward_step(simple_model, z, D)
        z_roundtrip = simple_model.forward(z_prev)
        assert torch.allclose(z_roundtrip, z, atol=1e-8), (
            f"Roundtrip failed: max diff {(z_roundtrip - z).abs().max():.2e}"
        )

    def test_backward_shape(self, simple_model):
        """Backward step returns same shape as input."""
        from dynamic.analysis.backtracking import backward_step

        z = torch.tensor([0.5, 0.3, 0.8])
        D = simple_model.get_D(z)
        z_prev = backward_step(simple_model, z, D)
        assert z_prev.shape == z.shape

    def test_backward_positive_region(self, simple_model):
        """Backward step in all-positive region."""
        from dynamic.analysis.backtracking import backward_step

        z = torch.tensor([1.0, 1.0, 1.0])
        D = torch.eye(3)  # all positive
        z_prev = backward_step(simple_model, z, D)
        z_fwd = simple_model.forward(z_prev)
        assert torch.allclose(z_fwd, z, atol=1e-8)


class TestVerifyForward:
    """Tests for forward verification."""

    def test_verify_true_for_correct(self, simple_model):
        """Correct inverse verifies successfully."""
        from dynamic.analysis.backtracking import verify_forward

        z0 = torch.tensor([0.5, 0.3, 0.8])
        z1 = simple_model.forward(z0)
        assert verify_forward(simple_model, z0, z1, tol=1e-6)

    def test_verify_false_for_incorrect(self, simple_model):
        """Random point fails verification."""
        from dynamic.analysis.backtracking import verify_forward

        z_wrong = torch.randn(3)
        z_target = torch.tensor([0.5, 0.3, 0.8])
        assert not verify_forward(simple_model, z_wrong, z_target, tol=1e-6)


class TestBitflipSearch:
    """Tests for hierarchical bitflip search."""

    def test_bitflip_finds_neighbor(self, simple_model):
        """When current D fails, bitflip search finds correct region."""
        from dynamic.analysis.backtracking import try_bitflips

        # Forward from a known z_prev to get z_target
        z_prev = torch.tensor([0.5, -0.3, 0.8])  # mixed signs
        z_target = simple_model.forward(z_prev)

        # Try bitflips starting from wrong region (all-positive)
        z_candidate = torch.ones(3)  # intentionally wrong
        result = try_bitflips(simple_model, z_target, z_candidate)
        assert result is not None, "Bitflip search should find a valid inverse"
        z_verify = simple_model.forward(result)
        assert torch.allclose(z_verify, z_target, atol=1e-6)


class TestBacktrackTrajectory:
    """Tests for full backward trajectory."""

    def test_backtrack_trajectory_shape(self, simple_model):
        """Backward trajectory has shape (T+1, M)."""
        from dynamic.analysis.backtracking import backtrack_trajectory

        z_T = torch.tensor([0.5, 0.3, 0.8])
        traj = backtrack_trajectory(simple_model, z_T, T=5)
        assert traj.shape == (6, 3)

    def test_backtrack_trajectory_last_is_z_T(self, simple_model):
        """Last state in backward trajectory is z_T."""
        from dynamic.analysis.backtracking import backtrack_trajectory

        z_T = torch.tensor([0.5, 0.3, 0.8])
        traj = backtrack_trajectory(simple_model, z_T, T=5)
        assert torch.allclose(traj[-1], z_T)

    def test_backtrack_trajectory_consistency(self, simple_model):
        """Each pair (z_{t-1}, z_t) satisfies F(z_{t-1}) ≈ z_t."""
        from dynamic.analysis.backtracking import backtrack_trajectory

        z_T = torch.tensor([0.5, 0.3, 0.8])
        traj = backtrack_trajectory(simple_model, z_T, T=5)
        for t in range(len(traj) - 1):
            z_fwd = simple_model.forward(traj[t])
            assert torch.allclose(z_fwd, traj[t + 1], atol=1e-6), (
                f"Step {t}: F(z_{t}) != z_{t+1}, "
                f"diff={( z_fwd - traj[t + 1]).abs().max():.2e}"
            )
