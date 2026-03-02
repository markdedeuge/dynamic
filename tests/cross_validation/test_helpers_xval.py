"""Cross-validation: helper-level deterministic comparisons.

These tests call the same helper function in both Julia and Python
with identical inputs and assert exact numerical equivalence.
"""

import numpy as np
import pytest
import torch

from tests.cross_validation.julia_wrapper import JuliaScyfi

pytestmark = pytest.mark.julia


@pytest.fixture()
def jw(jl_session):
    """Julia SCYFI wrapper."""
    return JuliaScyfi(jl_session)


# -- Float64 parameter fixtures for deterministic comparisons ----------
@pytest.fixture()
def params_2d():
    A = torch.tensor([[0.5, 0.0], [0.0, -0.3]], dtype=torch.float64)
    W = torch.tensor([[0.0, -0.6], [0.5, 0.0]], dtype=torch.float64)
    h = torch.tensor([0.37, -0.98], dtype=torch.float64)
    return A, W, h


# ======================================================================
# construct_relu_matrix
# ======================================================================
class TestConstructReluMatrixXval:
    """Bit-exact D matrix comparison."""

    @pytest.mark.parametrize("quadrant", [0, 1, 2, 3])
    def test_quadrant_2d(self, jw, quadrant):
        dim = 2
        jl_result = jw.construct_relu_matrix(quadrant, dim)
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix
        py_result = construct_relu_matrix(quadrant, dim).double()
        assert torch.allclose(py_result, jl_result, atol=1e-12)

    @pytest.mark.parametrize("quadrant", [0, 5, 7])
    def test_quadrant_3d(self, jw, quadrant):
        dim = 3
        jl_result = jw.construct_relu_matrix(quadrant, dim)
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix
        py_result = construct_relu_matrix(quadrant, dim).double()
        assert torch.allclose(py_result, jl_result, atol=1e-12)


# ======================================================================
# Factor computation
# ======================================================================
class TestFactorComputationXval:
    """z-factor and h-factor must match exactly for given D-list."""

    def _make_d_list(self, quadrants, dim):
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix
        return torch.stack(
            [construct_relu_matrix(q, dim).double() for q in quadrants]
        )

    def test_z_factor_order1(self, jw, params_2d):
        A, W, _ = params_2d
        D_list = self._make_d_list([3], 2)
        from dynamic.analysis.scyfi_helpers import get_factor_in_front_of_z
        py = get_factor_in_front_of_z(A, W, D_list, 1)
        jl = jw.get_factor_in_front_of_z(A, W, D_list, 1)
        assert torch.allclose(py, jl, atol=1e-12)

    def test_z_factor_order3(self, jw, params_2d):
        A, W, _ = params_2d
        D_list = self._make_d_list([1, 2, 3], 2)
        from dynamic.analysis.scyfi_helpers import get_factor_in_front_of_z
        py = get_factor_in_front_of_z(A, W, D_list, 3)
        jl = jw.get_factor_in_front_of_z(A, W, D_list, 3)
        assert torch.allclose(py, jl, atol=1e-12)

    def test_h_factor_order1(self, jw, params_2d):
        A, W, _ = params_2d
        D_list = self._make_d_list([3], 2)
        from dynamic.analysis.scyfi_helpers import get_factor_in_front_of_h
        py = get_factor_in_front_of_h(A, W, D_list, 1)
        jl = jw.get_factor_in_front_of_h(A, W, D_list, 1)
        assert torch.allclose(py, jl, atol=1e-12)

    def test_h_factor_order3(self, jw, params_2d):
        A, W, _ = params_2d
        D_list = self._make_d_list([1, 2, 3], 2)
        from dynamic.analysis.scyfi_helpers import get_factor_in_front_of_h
        py = get_factor_in_front_of_h(A, W, D_list, 3)
        jl = jw.get_factor_in_front_of_h(A, W, D_list, 3)
        assert torch.allclose(py, jl, atol=1e-12)


# ======================================================================
# Cycle point candidate
# ======================================================================
class TestCyclePointCandidateXval:
    """Candidate z* must match for given D-list."""

    def _make_d_list(self, quadrants, dim):
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix
        return torch.stack(
            [construct_relu_matrix(q, dim).double() for q in quadrants]
        )

    def test_candidate_order1(self, jw, params_2d):
        A, W, h = params_2d
        D_list = self._make_d_list([3], 2)
        from dynamic.analysis.scyfi_helpers import get_cycle_point_candidate
        py = get_cycle_point_candidate(A, W, D_list, h, 1)
        jl = jw.get_cycle_point_candidate(A, W, D_list, h, 1)
        if jl is not None and py is not None:
            assert torch.allclose(py, jl, atol=1e-10)

    def test_candidate_order2(self, jw, params_2d):
        A, W, h = params_2d
        D_list = self._make_d_list([1, 2], 2)
        from dynamic.analysis.scyfi_helpers import get_cycle_point_candidate
        py = get_cycle_point_candidate(A, W, D_list, h, 2)
        jl = jw.get_cycle_point_candidate(A, W, D_list, h, 2)
        if jl is not None and py is not None:
            assert torch.allclose(py, jl, atol=1e-10)


# ======================================================================
# Eigenvalues
# ======================================================================
class TestEigvalsXval:
    """Eigenvalue magnitudes must match for given D-list."""

    def _make_d_list(self, quadrants, dim):
        from dynamic.analysis.scyfi_helpers import construct_relu_matrix
        return torch.stack(
            [construct_relu_matrix(q, dim).double() for q in quadrants]
        )

    def test_eigvals_order1(self, jw, params_2d):
        A, W, _ = params_2d
        D_list = self._make_d_list([3], 2)
        from dynamic.analysis.scyfi_helpers import get_eigvals
        py = get_eigvals(A, W, D_list, 1)
        jl = jw.get_eigvals(A, W, D_list, 1)
        np.testing.assert_allclose(
            np.sort(np.abs(py)), np.sort(np.abs(jl)), atol=1e-10
        )

    def test_eigvals_order2(self, jw, params_2d):
        A, W, _ = params_2d
        D_list = self._make_d_list([1, 3], 2)
        from dynamic.analysis.scyfi_helpers import get_eigvals
        py = get_eigvals(A, W, D_list, 2)
        jl = jw.get_eigvals(A, W, D_list, 2)
        np.testing.assert_allclose(
            np.sort(np.abs(py)), np.sort(np.abs(jl)), atol=1e-10
        )


# ======================================================================
# Latent step and time series
# ======================================================================
class TestLatentStepXval:
    """Single PLRNN step must match exactly."""

    def test_step(self, jw, params_2d):
        A, W, h = params_2d
        z = torch.tensor([1.0, -0.5], dtype=torch.float64)
        from dynamic.analysis.scyfi_helpers import latent_step
        py = latent_step(z, A, W, h)
        jl = jw.latent_step(z, A, W, h)
        assert torch.allclose(py, jl, atol=1e-12)

    def test_time_series_3_steps(self, jw, params_2d):
        A, W, h = params_2d
        z0 = torch.tensor([0.5, -0.3], dtype=torch.float64)
        from dynamic.analysis.scyfi_helpers import get_latent_time_series
        py = get_latent_time_series(3, A, W, h, 2, z_0=z0)
        jl = jw.get_latent_time_series(3, A, W, h, 2, z_0=z0)
        for t in range(3):
            assert torch.allclose(py[t], jl[t], atol=1e-12)

    def test_time_series_10_steps(self, jw, params_2d):
        A, W, h = params_2d
        z0 = torch.tensor([2.0, -1.0], dtype=torch.float64)
        from dynamic.analysis.scyfi_helpers import get_latent_time_series
        py = get_latent_time_series(10, A, W, h, 2, z_0=z0)
        jl = jw.get_latent_time_series(10, A, W, h, 2, z_0=z0)
        for t in range(10):
            assert torch.allclose(py[t], jl[t], atol=1e-10)
