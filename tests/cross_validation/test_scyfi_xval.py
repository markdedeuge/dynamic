"""Cross-validation: algorithm-level comparisons.

Since both implementations use random initialisation, we cannot compare
trajectory-by-trajectory. Instead we verify:
  1. Both find the same NUMBER of cycles at each order.
  2. All cycles found by each implementation are self-consistent.
  3. Eigenvalue sets are compatible (sorted magnitudes match).
"""

import numpy as np
import pytest
import torch

from tests.cross_validation.julia_wrapper import JuliaScyfi

pytestmark = pytest.mark.julia


@pytest.fixture()
def jw(jl_session):
    return JuliaScyfi(jl_session)


# ======================================================================
# PLRNN fixed points
# ======================================================================
class TestFixedPointsXval:
    """Compare FP counts and self-consistency."""

    def test_fp_count_2d_holes(self, jw, system_2d_holes):
        """Both find 1 FP in the 2D holes system."""
        A, W, h = system_2d_holes
        from dynamic.analysis.scyfi import find_cycles

        py_result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )
        jl_result = jw.find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )
        py_count = len(py_result[0][0])
        jl_count = len(jl_result[0][0])
        assert py_count == jl_count, (
            f"FP count mismatch: Python={py_count}, Julia={jl_count}"
        )

    def test_fp_count_2d(self, jw, system_2d_fp):
        """Both find 1 FP in the 2D FP system."""
        A, W, h = system_2d_fp
        from dynamic.analysis.scyfi import find_cycles

        py_result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )
        jl_result = jw.find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )
        py_count = len(py_result[0][0])
        jl_count = len(jl_result[0][0])
        assert py_count == jl_count

    def test_fp_self_consistency_both(self, jw, system_2d_fp):
        """All FPs found by both are valid fixed points."""
        A, W, h = system_2d_fp
        from dynamic.analysis.scyfi import find_cycles
        from dynamic.analysis.scyfi_helpers import latent_step

        py_result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )
        jl_result = jw.find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )

        # Verify Python FPs
        for traj in py_result[0][0]:
            z_star = traj[0]
            z_next = latent_step(z_star, A, W, h)
            assert torch.norm(z_next - z_star).item() < 1e-6

        # Verify Julia FPs using Python latent_step
        for traj in jl_result[0][0]:
            z_star = traj[0]
            z_next = latent_step(z_star, A, W, h)
            assert torch.norm(z_next - z_star).item() < 1e-6

    def test_fp_values_close(self, jw, system_2d_fp):
        """FP values from both implementations are close."""
        A, W, h = system_2d_fp
        from dynamic.analysis.scyfi import find_cycles

        py_result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )
        jl_result = jw.find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )

        if py_result[0][0] and jl_result[0][0]:
            py_fp = py_result[0][0][0][0]
            jl_fp = jl_result[0][0][0][0]
            assert torch.allclose(py_fp, jl_fp, atol=1e-6)


# ======================================================================
# PLRNN cycles
# ======================================================================
class TestCyclesXval:
    """Compare cycle counts and properties."""

    def test_16_cycle_count(self, jw, system_2d_16cycle):
        """Both find the same number of 16-cycles."""
        A, W, h = system_2d_16cycle
        from dynamic.analysis.scyfi import find_cycles

        py_result = find_cycles(
            A, W, h, 16,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )
        jl_result = jw.find_cycles(
            A, W, h, 16,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )
        py_count = len(py_result[0][15])
        jl_count = len(jl_result[0][15])
        assert py_count == jl_count, (
            f"16-cycle count: Python={py_count}, Julia={jl_count}"
        )

    def test_cycle_self_consistency(self, jw, system_2d_16cycle):
        """All cycles from both are self-consistent (F^k(z*) = z*)."""
        A, W, h = system_2d_16cycle
        from dynamic.analysis.scyfi import find_cycles
        from dynamic.analysis.scyfi_helpers import latent_step

        py_result = find_cycles(
            A, W, h, 16,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )
        jl_result = jw.find_cycles(
            A, W, h, 16,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )

        # Verify Python cycles
        for traj in py_result[0][15]:
            z = traj[0]
            for _ in range(16):
                z = latent_step(z, A, W, h)
            assert torch.norm(z - traj[0]).item() < 1e-4

        # Verify Julia cycles
        for traj in jl_result[0][15]:
            z = traj[0]
            for _ in range(16):
                z = latent_step(z, A, W, h)
            assert torch.norm(z - traj[0]).item() < 1e-4

    def test_eigenvalue_magnitudes_agree(self, jw, system_2d_16cycle):
        """Eigenvalue sorted magnitudes match between implementations."""
        A, W, h = system_2d_16cycle
        from dynamic.analysis.scyfi import find_cycles

        py_result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )
        jl_result = jw.find_cycles(
            A, W, h, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )

        if py_result[1][0] and jl_result[1][0]:
            py_eig = np.sort(np.abs(py_result[1][0][0]))
            jl_eig = np.sort(np.abs(jl_result[1][0][0]))
            np.testing.assert_allclose(py_eig, jl_eig, atol=1e-6)


# ======================================================================
# shPLRNN
# ======================================================================
class TestShPLRNNXval:
    """shPLRNN cross-validation."""

    def test_sh_fp_count(self, jw, sh_system_1fp):
        """Both find 1 FP in the shPLRNN system."""
        A, W1, W2, h1, h2 = sh_system_1fp
        from dynamic.analysis.scyfi import find_cycles_sh

        py_result = find_cycles_sh(
            A, W1, W2, h1, h2, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )
        jl_result = jw.find_cycles_sh(
            A, W1, W2, h1, h2, 1,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )
        py_count = len(py_result[0][0])
        jl_count = len(jl_result[0][0])
        assert py_count == jl_count, (
            f"shPLRNN FP count: Python={py_count}, Julia={jl_count}"
        )

    def test_sh_cycles_count(self, jw, sh_system_1fp):
        """Both find the same order-2 and order-4 cycle counts."""
        A, W1, W2, h1, h2 = sh_system_1fp
        from dynamic.analysis.scyfi import find_cycles_sh

        py_result = find_cycles_sh(
            A, W1, W2, h1, h2, 4,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )
        jl_result = jw.find_cycles_sh(
            A, W1, W2, h1, h2, 4,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )

        for order in range(4):
            py_count = len(py_result[0][order])
            jl_count = len(jl_result[0][order])
            assert py_count == jl_count, (
                f"Order {order + 1}: Python={py_count}, Julia={jl_count}"
            )
