"""TDD tests for PLRNN model architectures (Phase 1).

Tests written FIRST per TDD methodology. Models are in:
- dynamic.models.plrnn (base PLRNN)
- dynamic.models.shallow_plrnn (shPLRNN)
- dynamic.models.alrnn (ALRNN)
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# Base PLRNN tests
# ---------------------------------------------------------------------------
class TestPLRNN:
    """Tests for the base PLRNN: z_t = diag(A) z_{t-1} + W ReLU(z_{t-1}) + h."""

    @pytest.fixture()
    def model(self):
        from dynamic.models.plrnn import PLRNN

        torch.manual_seed(42)
        return PLRNN(M=4)

    def test_forward_shape_1d(self, model):
        """Forward pass on a single state returns shape (M,)."""
        z = torch.randn(4)
        out = model.forward(z)
        assert out.shape == (4,)

    def test_forward_shape_batched(self, model):
        """Forward pass on a batch returns shape (B, M)."""
        z = torch.randn(8, 4)
        out = model.forward(z)
        assert out.shape == (8, 4)

    def test_diagonal_A(self, model):
        """A is stored as a vector and applied element-wise (diagonal)."""
        assert model.A.shape == (4,), "A should be stored as a 1D vector"

    def test_W_off_diagonal(self, model):
        """W is a full M×M matrix (off-diagonal connectivity)."""
        assert model.W.shape == (4, 4)

    def test_bias_shape(self, model):
        """Bias h has shape (M,)."""
        assert model.h.shape == (4,)

    def test_get_D(self, model):
        """D(z) is a binary diagonal: d_i = 1 iff z_i > 0."""
        z = torch.tensor([1.0, -2.0, 0.5, -0.1])
        D = model.get_D(z)
        expected = torch.diag(torch.tensor([1.0, 0.0, 1.0, 0.0]))
        assert torch.allclose(D, expected)

    def test_get_D_zero_boundary(self, model):
        """z_i = 0 maps to d_i = 0 (ReLU convention: max(0, z))."""
        z = torch.tensor([0.0, 1.0, -1.0, 0.0])
        D = model.get_D(z)
        diag = torch.diag(D)
        assert diag[0] == 0.0
        assert diag[1] == 1.0
        assert diag[2] == 0.0
        assert diag[3] == 0.0

    def test_jacobian_formula(self, model):
        """Jacobian equals diag(A) + W·D(z)."""
        z = torch.tensor([1.0, -2.0, 0.5, -0.1])
        J = model.get_jacobian(z)
        D = model.get_D(z)
        expected = torch.diag(model.A) + model.W @ D
        assert torch.allclose(J, expected, atol=1e-6)

    def test_jacobian_finite_difference(self, model):
        """Jacobian matches finite-difference approximation."""
        z = torch.tensor([1.0, 2.0, 0.5, 3.0])  # all positive to avoid kink
        J = model.get_jacobian(z)
        eps = 1e-5
        J_fd = torch.zeros(4, 4)
        f0 = model.forward(z)
        for i in range(4):
            z_pert = z.clone()
            z_pert[i] += eps
            f1 = model.forward(z_pert)
            J_fd[:, i] = (f1 - f0) / eps
        assert torch.allclose(J, J_fd, atol=1e-4)

    def test_subregion_id_type(self, model):
        """Subregion ID is a hashable tuple of ints."""
        z = torch.randn(4)
        rid = model.get_subregion_id(z)
        assert isinstance(rid, tuple)
        assert all(isinstance(x, int) for x in rid)
        assert all(x in (0, 1) for x in rid)

    def test_subregion_id_deterministic(self, model):
        """Same z always gives same subregion ID."""
        z = torch.tensor([1.0, -1.0, 0.5, -0.5])
        assert model.get_subregion_id(z) == model.get_subregion_id(z)

    def test_forward_trajectory_shape(self, model):
        """Trajectory of length T has shape (T+1, M)."""
        z0 = torch.randn(4)
        traj = model.forward_trajectory(z0, T=20)
        assert traj.shape == (21, 4)

    def test_forward_trajectory_first_state(self, model):
        """First state in trajectory is z0."""
        z0 = torch.randn(4)
        traj = model.forward_trajectory(z0, T=10)
        assert torch.allclose(traj[0], z0)

    def test_forward_trajectory_consistency(self, model):
        """Each step in trajectory matches forward(z_t)."""
        z0 = torch.randn(4)
        traj = model.forward_trajectory(z0, T=5)
        for t in range(5):
            expected = model.forward(traj[t])
            assert torch.allclose(traj[t + 1], expected, atol=1e-6)

    def test_forward_deterministic(self, model):
        """Forward pass is deterministic (no stochasticity)."""
        z = torch.randn(4)
        a = model.forward(z)
        b = model.forward(z)
        assert torch.allclose(a, b)

    def test_pl_form_equivalence(self, model):
        """ReLU form equals PL form: (A + WD)z + h."""
        z = torch.tensor([1.5, -0.5, 2.0, -1.0])
        relu_out = model.forward(z)
        D = model.get_D(z)
        pl_out = (torch.diag(model.A) + model.W @ D) @ z + model.h
        assert torch.allclose(relu_out, pl_out, atol=1e-6)


# ---------------------------------------------------------------------------
# Shallow PLRNN tests
# ---------------------------------------------------------------------------
class TestShallowPLRNN:
    """Tests for shPLRNN: z_t = A z_{t-1} + W1 Φ(W2 z_{t-1} + h2) + h1."""

    @pytest.fixture()
    def model(self):
        from dynamic.models.shallow_plrnn import ShallowPLRNN

        torch.manual_seed(42)
        return ShallowPLRNN(M=3, H=10)

    def test_forward_shape(self, model):
        """Forward pass returns shape (M,)."""
        z = torch.randn(3)
        assert model.forward(z).shape == (3,)

    def test_forward_shape_batched(self, model):
        """Batched forward returns (B, M)."""
        z = torch.randn(5, 3)
        assert model.forward(z).shape == (5, 3)

    def test_parameter_shapes(self, model):
        """W1 is (M,H), W2 is (H,M), h2 is (H,)."""
        assert model.W1.shape == (3, 10)
        assert model.W2.shape == (10, 3)
        assert model.h2.shape == (10,)
        assert model.h1.shape == (3,)
        assert model.A.shape == (3,)

    def test_hidden_activation(self, model):
        """Hidden activations are ReLU of W2·z + h2."""
        z = torch.randn(3)
        hidden = model.W2 @ z + model.h2
        D_h = model.get_D_hidden(z)
        # D_h should be diagonal with 1s where hidden > 0
        expected_diag = (hidden > 0).float()
        assert torch.allclose(torch.diag(D_h), expected_diag)

    def test_jacobian_formula(self, model):
        """Jacobian is diag(A) + W1·D_h·W2."""
        z = torch.randn(3)
        J = model.get_jacobian(z)
        D_h = model.get_D_hidden(z)
        expected = torch.diag(model.A) + model.W1 @ D_h @ model.W2
        assert torch.allclose(J, expected, atol=1e-6)

    def test_subregion_id(self, model):
        """Subregion ID based on hidden layer activation pattern."""
        z = torch.randn(3)
        rid = model.get_subregion_id(z)
        assert isinstance(rid, tuple)
        assert len(rid) == 10  # H hidden units determine the region

    def test_trajectory(self, model):
        """Trajectory generation works with correct shape."""
        z0 = torch.randn(3)
        traj = model.forward_trajectory(z0, T=15)
        assert traj.shape == (16, 3)


# ---------------------------------------------------------------------------
# ALRNN tests
# ---------------------------------------------------------------------------
class TestALRNN:
    """Tests for ALRNN: only last P dims use ReLU."""

    @pytest.fixture()
    def model(self):
        from dynamic.models.alrnn import ALRNN

        torch.manual_seed(42)
        return ALRNN(M=10, P=3)

    def test_forward_shape(self, model):
        """Forward pass returns shape (M,)."""
        z = torch.randn(10)
        assert model.forward(z).shape == (10,)

    def test_forward_shape_batched(self, model):
        """Batched forward returns (B, M)."""
        z = torch.randn(4, 10)
        assert model.forward(z).shape == (4, 10)

    def test_linear_dims_always_pass(self, model):
        """First M-P dimensions always have d_i = 1 in D."""
        z = torch.tensor(
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, 1.0, -1.0, 0.5]
        )
        D = model.get_D(z)
        diag = torch.diag(D)
        # First 7 (M-P=10-3) dims always 1, regardless of sign
        assert torch.all(diag[:7] == 1.0)
        # Last 3 dims follow ReLU rule
        assert diag[7] == 1.0  # z[7] = 1.0 > 0
        assert diag[8] == 0.0  # z[8] = -1.0 < 0
        assert diag[9] == 1.0  # z[9] = 0.5 > 0

    def test_subregion_count(self, model):
        """ALRNN has 2^P possible subregions, not 2^M."""
        # With P=3, there are 8 possible subregion IDs
        z_all_pos = torch.ones(10)
        z_all_neg = -torch.ones(10)
        rid_pos = model.get_subregion_id(z_all_pos)
        rid_neg = model.get_subregion_id(z_all_neg)
        # Only last P bits should differ
        assert rid_pos[:7] == rid_neg[:7]  # linear dims unchanged
        assert len(rid_pos) == 10  # full M, but first M-P always 1

    def test_jacobian(self, model):
        """Jacobian A + WD with correct D for ALRNN."""
        z = torch.randn(10)
        J = model.get_jacobian(z)
        D = model.get_D(z)
        expected = torch.diag(model.A) + model.W @ D
        assert torch.allclose(J, expected, atol=1e-6)

    def test_trajectory(self, model):
        """Trajectory generation works."""
        z0 = torch.randn(10)
        traj = model.forward_trajectory(z0, T=10)
        assert traj.shape == (11, 10)

    def test_parameters(self, model):
        """Parameter shapes are correct."""
        assert model.A.shape == (10,)
        assert model.W.shape == (10, 10)
        assert model.h.shape == (10,)
