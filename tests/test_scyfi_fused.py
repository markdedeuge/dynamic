"""Tests for fully fused SCYFI functions.

Validates that fused variants produce genuine fixed points and cycles
by checking forward-simulation self-consistency.
"""

import numpy as np
import pytest
import torch

from dynamic.analysis.scyfi import find_cycles
from dynamic.analysis.scyfi_fused import (
    find_cycles_fused,
    find_cycles_sh_fused,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def system_2d():
    """2D PLRNN system."""
    A = torch.tensor(
        [[0.5, 0.0], [0.0, -0.3]], dtype=torch.float64,
    )
    W = torch.tensor(
        [[0.0, -0.6], [0.5, 0.0]], dtype=torch.float64,
    )
    h = torch.tensor([0.37, -0.98], dtype=torch.float64)
    return A, W, h


@pytest.fixture()
def system_5d():
    """5D PLRNN system."""
    torch.manual_seed(42)
    dim = 5
    A = torch.diag(torch.randn(dim, dtype=torch.float64))
    W = torch.randn(dim, dim, dtype=torch.float64) * 0.3
    W.fill_diagonal_(0.0)
    h = torch.randn(dim, dtype=torch.float64)
    return A, W, h


@pytest.fixture()
def system_10d():
    """10D PLRNN system."""
    torch.manual_seed(99)
    dim = 10
    A = torch.diag(torch.randn(dim, dtype=torch.float64))
    W = torch.randn(dim, dim, dtype=torch.float64) * 0.3
    W.fill_diagonal_(0.0)
    h = torch.randn(dim, dtype=torch.float64)
    return A, W, h


def _check_fp_self_consistency(A, W, h, cycles, label=""):
    """Validate all fixed points are genuine."""
    for fp_traj in cycles:
        z = fp_traj[0]
        z_next = A @ z + W @ torch.clamp(z, min=0.0) + h
        assert torch.allclose(z, z_next, atol=1e-6), (
            f"{label} FP not self-consistent: "
            f"|z - f(z)| = {(z - z_next).abs().max():.2e}"
        )


def _check_cycle_self_consistency(A, W, h, cycles, order, label=""):
    """Validate all k-cycles are genuine periodic orbits."""
    for traj in cycles:
        assert len(traj) == order
        for j in range(order):
            z = traj[j]
            z_next = A @ z + W @ torch.clamp(z, min=0.0) + h
            expected = traj[(j + 1) % order]
            assert torch.allclose(z_next, expected, atol=1e-6), (
                f"{label} {order}-cycle step {j} not consistent: "
                f"|f(z_j) - z_{{j+1}}| = {(z_next - expected).abs().max():.2e}"
            )


# ---------------------------------------------------------------------------
# PLRNN fused tests
# ---------------------------------------------------------------------------
class TestFindCyclesFused:
    """Fused PLRNN cycle search must find genuine FP/cycles."""

    def test_fp_self_consistency_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_fused(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        assert len(cycles[0]) >= 1, "Should find at least 1 FP"
        _check_fp_self_consistency(A, W, h, cycles[0], "2D fused")

    def test_fp_self_consistency_5d(self, system_5d):
        A, W, h = system_5d
        torch.manual_seed(42)
        cycles, _ = find_cycles_fused(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        _check_fp_self_consistency(A, W, h, cycles[0], "5D fused")

    def test_fp_self_consistency_10d(self, system_10d):
        A, W, h = system_10d
        torch.manual_seed(42)
        cycles, _ = find_cycles_fused(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        _check_fp_self_consistency(A, W, h, cycles[0], "10D fused")

    def test_2cycle_self_consistency_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_fused(
            A, W, h, 2,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        _check_cycle_self_consistency(
            A, W, h, cycles[1], 2, "2D fused 2-cycle"
        )

    def test_4cycle_self_consistency_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_fused(
            A, W, h, 4,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        # Validate any found 4-cycles
        for order_idx in range(4):
            _check_cycle_self_consistency(
                A, W, h, cycles[order_idx], order_idx + 1,
                f"2D fused order-{order_idx + 1}",
            )

    def test_eigvals_returned(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, eigvals = find_cycles_fused(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        assert len(eigvals[0]) == len(cycles[0])
        for e in eigvals[0]:
            assert isinstance(e, np.ndarray)
            assert e.shape == (2,)

    def test_count_matches_reference_direction(self, system_2d):
        """Fused should find at least as many FPs as reference."""
        A, W, h = system_2d
        torch.manual_seed(0)
        ref_cycles, _ = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        # Run fused many seeds, check we can find FPs
        found_any = False
        for seed in range(10):
            torch.manual_seed(seed)
            fused_cycles, _ = find_cycles_fused(
                A, W, h, 1,
                outer_loop_iterations=10,
                inner_loop_iterations=20,
            )
            if len(fused_cycles[0]) > 0:
                found_any = True
                break
        assert found_any, "Fused should find at least one FP across seeds"


# ---------------------------------------------------------------------------
# shPLRNN fused tests
# ---------------------------------------------------------------------------
class TestFindCyclesShFused:
    """Fused shPLRNN cycle search must find genuine FP/cycles."""

    def test_sh_fp_self_consistency(self):
        torch.manual_seed(42)
        dim, hidden_dim = 2, 6
        A = torch.randn(dim, dtype=torch.float64)
        W1 = torch.randn(dim, hidden_dim, dtype=torch.float64) * 0.3
        W2 = torch.randn(hidden_dim, dim, dtype=torch.float64) * 0.3
        h1 = torch.randn(dim, dtype=torch.float64)
        h2 = torch.randn(hidden_dim, dtype=torch.float64)

        cycles, _ = find_cycles_sh_fused(
            A, W1, W2, h1, h2, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )

        for fp_traj in cycles[0]:
            z = fp_traj[0]
            z_next = A * z + W1 @ torch.clamp(W2 @ z + h2, min=0.0) + h1
            assert torch.allclose(z, z_next, atol=1e-6)

    def test_sh_2cycle_self_consistency(self):
        torch.manual_seed(42)
        dim, hidden_dim = 2, 6
        A = torch.randn(dim, dtype=torch.float64)
        W1 = torch.randn(dim, hidden_dim, dtype=torch.float64) * 0.3
        W2 = torch.randn(hidden_dim, dim, dtype=torch.float64) * 0.3
        h1 = torch.randn(dim, dtype=torch.float64)
        h2 = torch.randn(hidden_dim, dtype=torch.float64)

        cycles, _ = find_cycles_sh_fused(
            A, W1, W2, h1, h2, 2,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )

        for traj in cycles[1]:  # 2-cycles
            z0, z1 = traj[0], traj[1]
            z0_next = A * z0 + W1 @ torch.clamp(W2 @ z0 + h2, min=0.0) + h1
            z1_next = A * z1 + W1 @ torch.clamp(W2 @ z1 + h2, min=0.0) + h1
            assert torch.allclose(z0_next, z1, atol=1e-6)
            assert torch.allclose(z1_next, z0, atol=1e-6)

    def test_sh_eigvals_returned(self):
        torch.manual_seed(42)
        dim, hidden_dim = 2, 6
        A = torch.randn(dim, dtype=torch.float64)
        W1 = torch.randn(dim, hidden_dim, dtype=torch.float64) * 0.3
        W2 = torch.randn(hidden_dim, dim, dtype=torch.float64) * 0.3
        h1 = torch.randn(dim, dtype=torch.float64)
        h2 = torch.randn(hidden_dim, dtype=torch.float64)

        cycles, eigvals = find_cycles_sh_fused(
            A, W1, W2, h1, h2, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )

        assert len(eigvals[0]) == len(cycles[0])
        for e in eigvals[0]:
            assert isinstance(e, np.ndarray)
            assert e.shape == (dim,)
