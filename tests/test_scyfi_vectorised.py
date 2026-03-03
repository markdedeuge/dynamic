"""Tests for vectorised SCYFI functions.

Validates that vectorised variants produce genuine fixed points and
cycles by checking forward-simulation self-consistency.
"""

import numpy as np
import pytest
import torch

from dynamic.analysis.scyfi_vectorised import (
    find_cycles_sh_vectorised,
    find_cycles_vectorised,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def system_2d():
    A = torch.tensor(
        [[0.5, 0.0], [0.0, -0.3]],
        dtype=torch.float64,
    )
    W = torch.tensor(
        [[0.0, -0.6], [0.5, 0.0]],
        dtype=torch.float64,
    )
    h = torch.tensor([0.37, -0.98], dtype=torch.float64)
    return A, W, h


@pytest.fixture()
def system_5d():
    torch.manual_seed(42)
    dim = 5
    A = torch.diag(torch.randn(dim, dtype=torch.float64))
    W = torch.randn(dim, dim, dtype=torch.float64) * 0.3
    W.fill_diagonal_(0.0)
    h = torch.randn(dim, dtype=torch.float64)
    return A, W, h


@pytest.fixture()
def system_10d():
    torch.manual_seed(99)
    dim = 10
    A = torch.diag(torch.randn(dim, dtype=torch.float64))
    W = torch.randn(dim, dim, dtype=torch.float64) * 0.3
    W.fill_diagonal_(0.0)
    h = torch.randn(dim, dtype=torch.float64)
    return A, W, h


def _check_fp_self_consistency(A, W, h, cycles, label=""):
    for fp_traj in cycles:
        z = fp_traj[0]
        z_next = A @ z + W @ torch.clamp(z, min=0.0) + h
        assert torch.allclose(z, z_next, atol=1e-6), (
            f"{label} FP not self-consistent: "
            f"|z - f(z)| = {(z - z_next).abs().max():.2e}"
        )


def _check_cycle_self_consistency(A, W, h, cycles, order, label=""):
    for traj in cycles:
        assert len(traj) == order
        for j in range(order):
            z = traj[j]
            z_next = A @ z + W @ torch.clamp(z, min=0.0) + h
            expected = traj[(j + 1) % order]
            assert torch.allclose(z_next, expected, atol=1e-6), (
                f"{label} {order}-cycle step {j} not consistent: "
                f"|f(z_j) - z_{{j+1}}| = "
                f"{(z_next - expected).abs().max():.2e}"
            )


# ---------------------------------------------------------------------------
# PLRNN vectorised tests
# ---------------------------------------------------------------------------
class TestFindCyclesVectorised:
    def test_fp_self_consistency_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
        )
        assert len(cycles[0]) >= 1, "Should find at least 1 FP"
        _check_fp_self_consistency(A, W, h, cycles[0], "2D vec")

    def test_fp_self_consistency_5d(self, system_5d):
        A, W, h = system_5d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
        )
        _check_fp_self_consistency(A, W, h, cycles[0], "5D vec")

    def test_fp_self_consistency_10d(self, system_10d):
        A, W, h = system_10d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
        )
        _check_fp_self_consistency(A, W, h, cycles[0], "10D vec")

    def test_2cycle_self_consistency(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            2,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
        )
        _check_cycle_self_consistency(A, W, h, cycles[1], 2, "2D vec 2-cycle")

    def test_4cycle_self_consistency(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            4,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
        )
        for order_idx in range(4):
            _check_cycle_self_consistency(
                A,
                W,
                h,
                cycles[order_idx],
                order_idx + 1,
                f"2D vec order-{order_idx + 1}",
            )

    def test_eigvals_returned(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, eigvals = find_cycles_vectorised(
            A,
            W,
            h,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
        )
        assert len(eigvals[0]) == len(cycles[0])
        for e in eigvals[0]:
            assert isinstance(e, np.ndarray)
            assert e.shape == (2,)

    def test_finds_fps_across_seeds(self, system_2d):
        """Vectorised should find FPs reliably across seeds."""
        A, W, h = system_2d
        found_any = False
        for seed in range(10):
            torch.manual_seed(seed)
            cycles, _ = find_cycles_vectorised(
                A,
                W,
                h,
                1,
                outer_loop_iterations=10,
                inner_loop_iterations=20,
                batch_size=64,
            )
            if len(cycles[0]) > 0:
                found_any = True
                break
        assert found_any


# ---------------------------------------------------------------------------
# shPLRNN vectorised tests
# ---------------------------------------------------------------------------
class TestFindCyclesShVectorised:
    def test_sh_fp_self_consistency(self):
        torch.manual_seed(42)
        dim, hidden_dim = 2, 6
        A = torch.randn(dim, dtype=torch.float64)
        W1 = torch.randn(dim, hidden_dim, dtype=torch.float64) * 0.3
        W2 = torch.randn(hidden_dim, dim, dtype=torch.float64) * 0.3
        h1 = torch.randn(dim, dtype=torch.float64)
        h2 = torch.randn(hidden_dim, dtype=torch.float64)

        cycles, _ = find_cycles_sh_vectorised(
            A,
            W1,
            W2,
            h1,
            h2,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=32,
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

        cycles, _ = find_cycles_sh_vectorised(
            A,
            W1,
            W2,
            h1,
            h2,
            2,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=32,
        )

        for traj in cycles[1]:
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

        cycles, eigvals = find_cycles_sh_vectorised(
            A,
            W1,
            W2,
            h1,
            h2,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=32,
        )

        assert len(eigvals[0]) == len(cycles[0])
        for e in eigvals[0]:
            assert isinstance(e, np.ndarray)
            assert e.shape == (dim,)


# ---------------------------------------------------------------------------
# Integration tests for optimisation flags
# ---------------------------------------------------------------------------
class TestOptimisationFlags:
    def test_fast_solve_fp(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
            fast_solve=True,
        )
        assert len(cycles[0]) >= 1
        _check_fp_self_consistency(A, W, h, cycles[0], "fast_solve")

    def test_use_table_fp(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
            use_table=True,
        )
        assert len(cycles[0]) >= 1
        _check_fp_self_consistency(A, W, h, cycles[0], "use_table")

    def test_float32_fp(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
            precision="float32",
        )
        assert len(cycles[0]) >= 1
        for fp_traj in cycles[0]:
            z = fp_traj[0].to(torch.float64)
            A64, W64, h64 = (
                A.to(torch.float64),
                W.to(torch.float64),
                h.to(torch.float64),
            )
            z_next = A64 @ z + W64 @ torch.clamp(z, min=0.0) + h64
            assert torch.allclose(z, z_next, atol=1e-4), (
                f"float32 FP not self-consistent: "
                f"|z - f(z)| = {(z - z_next).abs().max():.2e}"
            )

    def test_all_opts_combined(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            2,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=64,
            fast_solve=True,
            use_table=True,
            compiled=True,
        )
        _check_fp_self_consistency(A, W, h, cycles[0], "all_opts")
        _check_cycle_self_consistency(
            A,
            W,
            h,
            cycles[1],
            2,
            "all_opts 2-cycle",
        )

    def test_auto_batch_size(self, system_2d):
        """batch_size=None should auto-select and still find FPs."""
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=None,
        )
        assert len(cycles[0]) >= 1
        _check_fp_self_consistency(A, W, h, cycles[0], "auto_batch")

    def test_fast_solve_5d_cycle(self, system_5d):
        """fast_solve falls back to linalg.solve at dim=5."""
        A, W, h = system_5d
        torch.manual_seed(42)
        cycles, _ = find_cycles_vectorised(
            A,
            W,
            h,
            2,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batch_size=32,
            fast_solve=True,
        )
        _check_fp_self_consistency(A, W, h, cycles[0], "fast_solve 5D")
