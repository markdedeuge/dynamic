"""TDD tests for SCYFI core helper functions (Phase 4, Round 2).

Tests written FIRST per TDD methodology. Implementation target:
    dynamic.analysis.scyfi_helpers

These test the low-level mathematical operations that underpin the
SCYFI algorithm: factor computation, candidate solving, time series,
eigenvalue extraction, and ReLU matrix construction.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures: small 2D PLRNN parameters
# ---------------------------------------------------------------------------
@pytest.fixture()
def plrnn_2d_params():
    """Small 2D PLRNN system from Julia tests."""
    A = torch.tensor([[0.5, 0.0], [0.0, -0.3]])
    W = torch.tensor([[0.0, -0.6], [0.5, 0.0]])
    h = torch.tensor([0.37, -0.98])
    return A, W, h


@pytest.fixture()
def shplrnn_params():
    """Small M=2, H=3 shPLRNN system."""
    A = torch.tensor([0.9, -0.5])
    W1 = torch.randn(2, 3) * 0.5
    W2 = torch.randn(3, 2) * 0.5
    h1 = torch.tensor([0.1, -0.2])
    h2 = torch.tensor([0.3, -0.1, 0.5])
    return A, W1, W2, h1, h2


# ---------------------------------------------------------------------------
# ReLU matrix construction
# ---------------------------------------------------------------------------
class TestConstructReluMatrix:
    """Tests for construct_relu_matrix."""

    def test_shape(self):
        """Output is (dim, dim) diagonal."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix

        D = construct_relu_matrix(0, 4)
        assert D.shape == (4, 4)

    def test_identity_quadrant(self):
        """Quadrant 2^dim - 1 (all ones) → identity."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix

        dim = 3
        D = construct_relu_matrix(2**dim - 1, dim)
        assert torch.allclose(D, torch.eye(dim))

    def test_zero_quadrant(self):
        """Quadrant 0 (all zeros) → zero matrix."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix

        D = construct_relu_matrix(0, 4)
        assert torch.allclose(D, torch.zeros(4, 4))

    def test_is_diagonal(self):
        """Result is always a diagonal matrix."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix

        D = construct_relu_matrix(5, 4)  # 5 = 0101 in binary
        # Check off-diagonal is zero
        off_diag = D - torch.diag(torch.diag(D))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag))

    def test_specific_pattern(self):
        """Quadrant 5 (binary 101) for dim=3 → [1, 0, 1]."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix

        D = construct_relu_matrix(5, 3)
        expected_diag = torch.tensor([1.0, 0.0, 1.0])
        assert torch.allclose(torch.diag(D), expected_diag)


class TestConstructReluMatrixList:
    """Tests for construct_relu_matrix_list."""

    def test_shape(self):
        """Output shape is (order, dim, dim)."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix_list

        D_list = construct_relu_matrix_list(4, 3)
        assert D_list.shape == (3, 4, 4)

    def test_each_is_diagonal(self):
        """Each matrix in the list is diagonal."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix_list

        D_list = construct_relu_matrix_list(3, 5)
        for i in range(5):
            D = D_list[i]
            off_diag = D - torch.diag(torch.diag(D))
            assert torch.allclose(off_diag, torch.zeros_like(off_diag))

    def test_binary_values(self):
        """Diagonal entries are 0 or 1."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix_list

        D_list = construct_relu_matrix_list(4, 10)
        for i in range(10):
            diag = torch.diag(D_list[i])
            assert torch.all((diag == 0.0) | (diag == 1.0))


# ---------------------------------------------------------------------------
# Factor computation (PLRNN)
# ---------------------------------------------------------------------------
class TestFactorComputation:
    """Tests for get_factor_in_front_of_z and get_factor_in_front_of_h."""

    def test_factor_z_order1(self, plrnn_2d_params):
        """Order 1: z-factor is just A + W·D₁."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_factor_in_front_of_z,
        )

        A, W, _ = plrnn_2d_params
        D = construct_relu_matrix(1, 2)  # D = [[0,0],[0,1]]
        D_list = D.unsqueeze(0)  # shape (1, 2, 2)

        result = get_factor_in_front_of_z(A, W, D_list, 1)
        expected = A + W @ D
        assert torch.allclose(result, expected, atol=1e-6)

    def test_factor_z_order2(self, plrnn_2d_params):
        """Order 2: z-factor is (A+W·D₂)(A+W·D₁)."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_factor_in_front_of_z,
        )

        A, W, _ = plrnn_2d_params
        D1 = construct_relu_matrix(1, 2)
        D2 = construct_relu_matrix(2, 2)
        D_list = torch.stack([D1, D2])  # shape (2, 2, 2)

        result = get_factor_in_front_of_z(A, W, D_list, 2)
        expected = (A + W @ D2) @ (A + W @ D1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_factor_h_order1(self, plrnn_2d_params):
        """Order 1: h-factor is identity."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_factor_in_front_of_h,
        )

        A, W, _ = plrnn_2d_params
        D = construct_relu_matrix(1, 2)
        D_list = D.unsqueeze(0)

        result = get_factor_in_front_of_h(A, W, D_list, 1)
        expected = torch.eye(2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_factor_h_order2(self, plrnn_2d_params):
        """Order 2: h-factor is (A+W·D₂) + I."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_factor_in_front_of_h,
        )

        A, W, _ = plrnn_2d_params
        D1 = construct_relu_matrix(1, 2)
        D2 = construct_relu_matrix(2, 2)
        # D_list passed to h-factor is D_list[1:] (indices 2..end) per Julia
        D_list = torch.stack([D1, D2])

        result = get_factor_in_front_of_h(A, W, D_list, 2)
        expected = (A + W @ D2) + torch.eye(2)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Cycle point candidate
# ---------------------------------------------------------------------------
class TestCyclePointCandidate:
    """Tests for get_cycle_point_candidate."""

    def test_invertible_returns_tensor(self, plrnn_2d_params):
        """Solvable system returns a Tensor."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_cycle_point_candidate,
        )

        A, W, h = plrnn_2d_params
        D = construct_relu_matrix(3, 2)  # all-ones
        D_list = D.unsqueeze(0)

        z = get_cycle_point_candidate(A, W, D_list, h, 1)
        assert z is not None
        assert isinstance(z, torch.Tensor)
        assert z.shape == (2,)

    def test_singular_returns_none(self):
        """Singular system returns None."""
        from dynamic.analysis.scyfi_helpers import get_cycle_point_candidate

        # Construct I - (A + WD) = 0 → singular
        A = torch.eye(2)
        W = torch.zeros(2, 2)
        h = torch.ones(2)
        D = torch.eye(2)
        D_list = D.unsqueeze(0)

        # A + W·D = I → I - I = 0 → singular
        z = get_cycle_point_candidate(A, W, D_list, h, 1)
        assert z is None


# ---------------------------------------------------------------------------
# Latent step and time series
# ---------------------------------------------------------------------------
class TestLatentStep:
    """Tests for single PLRNN step functions."""

    def test_plrnn_step(self, plrnn_2d_params):
        """Single step matches A*z + W*relu(z) + h."""
        from dynamic.analysis.scyfi_helpers import latent_step

        A, W, h = plrnn_2d_params
        z = torch.tensor([1.0, -0.5])

        result = latent_step(z, A, W, h)
        expected = A @ z + W @ torch.clamp(z, min=0.0) + h
        assert torch.allclose(result, expected, atol=1e-6)

    def test_sh_step(self, shplrnn_params):
        """Single shPLRNN step matches A.*z + W₁*relu(W₂*z + h₂) + h₁."""
        from dynamic.analysis.scyfi_helpers import latent_step_sh

        A, W1, W2, h1, h2 = shplrnn_params
        z = torch.tensor([0.5, -0.3])

        result = latent_step_sh(z, A, W1, W2, h1, h2)
        expected = A * z + W1 @ torch.clamp(W2 @ z + h2, min=0.0) + h1
        assert torch.allclose(result, expected, atol=1e-6)


class TestLatentTimeSeries:
    """Tests for get_latent_time_series."""

    def test_length(self, plrnn_2d_params):
        """Returns list of `order` tensors."""
        from dynamic.analysis.scyfi_helpers import get_latent_time_series

        A, W, h = plrnn_2d_params
        z0 = torch.tensor([0.5, -0.3])
        traj = get_latent_time_series(5, A, W, h, 2, z_0=z0)
        assert len(traj) == 5

    def test_first_state(self, plrnn_2d_params):
        """First state in trajectory is z0."""
        from dynamic.analysis.scyfi_helpers import get_latent_time_series

        A, W, h = plrnn_2d_params
        z0 = torch.tensor([0.5, -0.3])
        traj = get_latent_time_series(3, A, W, h, 2, z_0=z0)
        assert torch.allclose(traj[0], z0)

    def test_consistency(self, plrnn_2d_params):
        """Each step equals latent_step of previous."""
        from dynamic.analysis.scyfi_helpers import (
            get_latent_time_series,
            latent_step,
        )

        A, W, h = plrnn_2d_params
        z0 = torch.tensor([0.5, -0.3])
        traj = get_latent_time_series(4, A, W, h, 2, z_0=z0)
        for t in range(len(traj) - 1):
            expected = latent_step(traj[t], A, W, h)
            assert torch.allclose(traj[t + 1], expected, atol=1e-6)

    def test_sh_length(self, shplrnn_params):
        """shPLRNN time series returns correct length."""
        from dynamic.analysis.scyfi_helpers import get_latent_time_series_sh

        A, W1, W2, h1, h2 = shplrnn_params
        z0 = torch.tensor([0.5, -0.3])
        traj = get_latent_time_series_sh(3, A, W1, W2, h1, h2, 2, z_0=z0)
        assert len(traj) == 3


# ---------------------------------------------------------------------------
# Eigenvalues
# ---------------------------------------------------------------------------
class TestEigvals:
    """Tests for eigenvalue computation."""

    def test_order1(self, plrnn_2d_params):
        """Order 1 eigenvalues match numpy eigvals of A + W·D."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_eigvals,
        )

        A, W, _ = plrnn_2d_params
        D = construct_relu_matrix(3, 2)
        D_list = D.unsqueeze(0)

        result = get_eigvals(A, W, D_list, 1)
        expected = np.linalg.eigvals((A + W @ D).numpy())
        np.testing.assert_allclose(np.sort(np.abs(result)),
                                   np.sort(np.abs(expected)),
                                   atol=1e-6)

    def test_order2(self, plrnn_2d_params):
        """Order 2 eigenvalues match composed matrix eigvals."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_eigvals,
        )

        A, W, _ = plrnn_2d_params
        D1 = construct_relu_matrix(1, 2)
        D2 = construct_relu_matrix(2, 2)
        D_list = torch.stack([D1, D2])

        result = get_eigvals(A, W, D_list, 2)
        composed = ((A + W @ D2) @ (A + W @ D1)).numpy()
        expected = np.linalg.eigvals(composed)
        np.testing.assert_allclose(np.sort(np.abs(result)),
                                   np.sort(np.abs(expected)),
                                   atol=1e-6)


# ---------------------------------------------------------------------------
# Loop iteration defaults
# ---------------------------------------------------------------------------
class TestSetLoopIterations:
    """Tests for set_loop_iterations."""

    def test_defaults_low_order(self):
        """None → default values for low order."""
        from dynamic.analysis.scyfi_helpers import set_loop_iterations

        outer, inner = set_loop_iterations(1, None, None)
        assert outer > 0
        assert inner > 0

    def test_override(self):
        """Explicit values passed through unchanged."""
        from dynamic.analysis.scyfi_helpers import set_loop_iterations

        outer, inner = set_loop_iterations(5, 42, 99)
        assert outer == 42
        assert inner == 99

    def test_higher_order_more_iterations(self):
        """Higher order gets more iterations than lower order."""
        from dynamic.analysis.scyfi_helpers import set_loop_iterations

        _, inner_low = set_loop_iterations(2, None, None)
        _, inner_high = set_loop_iterations(10, None, None)
        assert inner_high >= inner_low


# ---------------------------------------------------------------------------
# shPLRNN-specific helpers
# ---------------------------------------------------------------------------
class TestShPLRNNHelpers:
    """Tests for shPLRNN factor computation and pool construction."""

    def test_factors_sh_order1(self, shplrnn_params):
        """shPLRNN order-1 z-factor matches diag(A) + W₁·D·W₂."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_factors_sh,
        )

        A, W1, W2, _, _ = shplrnn_params
        D = construct_relu_matrix(5, 3)  # H=3
        D_list = D.unsqueeze(0)

        z_factor, _, _ = get_factors_sh(A, W1, W2, D_list, 1)
        expected = torch.diag(A) + W1 @ D @ W2
        assert torch.allclose(z_factor, expected, atol=1e-6)

    def test_candidate_sh(self, shplrnn_params):
        """shPLRNN candidate returns a tensor for solvable system."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_cycle_point_candidate_sh,
        )

        A, W1, W2, h1, h2 = shplrnn_params
        D = construct_relu_matrix(5, 3)
        D_list = D.unsqueeze(0)

        z = get_cycle_point_candidate_sh(A, W1, W2, h1, h2, D_list, 1)
        # May or may not be solvable, but should not error
        if z is not None:
            assert z.shape == (2,)

    def test_relu_pool_shape(self, shplrnn_params):
        """Pool has (N, H, H) shape with N ≤ 2^H."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix_pool

        A, W1, W2, h1, h2 = shplrnn_params
        pool = construct_relu_matrix_pool(A, W1, W2, h1, h2, 2, 3)
        assert pool.ndim == 3
        assert pool.shape[1] == 3  # H
        assert pool.shape[2] == 3  # H
        assert pool.shape[0] <= 2**3  # at most 2^H unique patterns

    def test_relu_pool_diagonal(self, shplrnn_params):
        """All D's in pool are diagonal matrices."""
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix_pool

        A, W1, W2, h1, h2 = shplrnn_params
        pool = construct_relu_matrix_pool(A, W1, W2, h1, h2, 2, 3)
        for i in range(pool.shape[0]):
            D = pool[i]
            off_diag = D - torch.diag(torch.diag(D))
            assert torch.allclose(off_diag, torch.zeros_like(off_diag))

    def test_eigvals_sh(self, shplrnn_params):
        """shPLRNN eigenvalues match composed matrix eigvals."""
        from dynamic.analysis.scyfi_helpers import (
            construct_relu_matrix,
            get_eigvals_sh,
        )

        A, W1, W2, _, _ = shplrnn_params
        D = construct_relu_matrix(5, 3)
        D_list = D.unsqueeze(0)

        result = get_eigvals_sh(A, W1, W2, D_list, 1)
        composed = (torch.diag(A) + W1 @ D @ W2).numpy()
        expected = np.linalg.eigvals(composed)
        np.testing.assert_allclose(
            np.sort(np.abs(result)), np.sort(np.abs(expected)), atol=1e-6
        )
