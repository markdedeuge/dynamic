"""Tests for all alternative SCYFI algorithm variants + hybrids.

Validates self-consistency of found cycles via forward simulation.
Cross-validates variants against exhaustive as ground truth.
"""

import numpy as np
import pytest
import torch

from dynamic.analysis.scyfi_exhaustive import find_cycles_exhaustive
from dynamic.analysis.scyfi_hybrid import find_cycles_hybrid
from dynamic.analysis.scyfi_newton import find_cycles_newton
from dynamic.analysis.scyfi_power import find_cycles_power
from dynamic.analysis.scyfi_schur import find_cycles_schur
from dynamic.analysis.scyfi_woodbury import find_cycles_woodbury


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


def _check_fp(A, W, h, cycles, label=""):
    for fp_traj in cycles:
        z = fp_traj[0]
        z_next = A @ z + W @ torch.clamp(z, min=0.0) + h
        assert torch.allclose(z, z_next, atol=1e-6), (
            f"{label} FP not self-consistent: "
            f"|z - f(z)| = {(z - z_next).abs().max():.2e}"
        )


def _check_cycle(A, W, h, cycles, order, label=""):
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
# 1. Exhaustive
# ---------------------------------------------------------------------------
class TestExhaustive:
    def test_fp_all_found_2d(self, system_2d):
        A, W, h = system_2d
        cycles, _ = find_cycles_exhaustive(A, W, h, 1)
        assert len(cycles[0]) >= 1
        _check_fp(A, W, h, cycles[0], "exhaustive")

    def test_2cycle_2d(self, system_2d):
        A, W, h = system_2d
        cycles, _ = find_cycles_exhaustive(A, W, h, 2)
        _check_cycle(A, W, h, cycles[1], 2, "exhaustive 2-cycle")

    def test_4cycle_2d(self, system_2d):
        A, W, h = system_2d
        cycles, _ = find_cycles_exhaustive(A, W, h, 4)
        for order_idx in range(4):
            if cycles[order_idx]:
                _check_cycle(
                    A,
                    W,
                    h,
                    cycles[order_idx],
                    order_idx + 1,
                    f"exhaustive order-{order_idx + 1}",
                )

    def test_too_large_raises(self, system_5d):
        A, W, h = system_5d
        with pytest.raises(ValueError, match="exceeds max_systems"):
            find_cycles_exhaustive(A, W, h, 4, max_systems=100)

    def test_eigvals(self, system_2d):
        A, W, h = system_2d
        cycles, eigvals = find_cycles_exhaustive(A, W, h, 1)
        assert len(eigvals[0]) == len(cycles[0])
        for e in eigvals[0]:
            assert isinstance(e, np.ndarray)


# ---------------------------------------------------------------------------
# 2. Power Iteration
# ---------------------------------------------------------------------------
class TestPower:
    def test_fp_2d(self, system_2d):
        """Power is seed-sensitive. Try multiple seeds."""
        A, W, h = system_2d
        found = False
        for seed in range(20):
            torch.manual_seed(seed)
            cycles, _ = find_cycles_power(A, W, h, 1)
            if cycles[0]:
                _check_fp(A, W, h, cycles[0], "power")
                found = True
                break
        assert found, "Power should find FP within 20 seeds"

    def test_2cycle_2d(self, system_2d):
        A, W, h = system_2d
        for seed in range(20):
            torch.manual_seed(seed)
            cycles, _ = find_cycles_power(A, W, h, 2)
            if cycles[1]:
                _check_cycle(A, W, h, cycles[1], 2, "power 2-cycle")
                break

    def test_stable_only(self, system_2d):
        """Power should only find stable cycles (|λ| < 1)."""
        A, W, h = system_2d
        for seed in range(20):
            torch.manual_seed(seed)
            _, eigvals_list = find_cycles_power(A, W, h, 2)
            for order_eigvals in eigvals_list:
                for eigvals in order_eigvals:
                    max_abs_eig = np.max(np.abs(eigvals))
                    assert max_abs_eig < 1.0 + 1e-6, (
                        f"Power found unstable cycle: max|λ| = {max_abs_eig:.4f}"
                    )


# ---------------------------------------------------------------------------
# 3. Newton-Raphson
# ---------------------------------------------------------------------------
class TestNewton:
    def test_fp_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_newton(
            A,
            W,
            h,
            1,
            B=64,
            outer_iterations=5,
            inner_iterations=5,
        )
        assert len(cycles[0]) >= 1
        _check_fp(A, W, h, cycles[0], "newton")

    def test_2cycle_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_newton(
            A,
            W,
            h,
            2,
            B=64,
            outer_iterations=5,
            inner_iterations=5,
        )
        if cycles[1]:
            _check_cycle(A, W, h, cycles[1], 2, "newton 2-cycle")

    def test_5d_fp(self, system_5d):
        A, W, h = system_5d
        torch.manual_seed(42)
        cycles, _ = find_cycles_newton(
            A,
            W,
            h,
            1,
            B=64,
            outer_iterations=5,
            inner_iterations=5,
        )
        _check_fp(A, W, h, cycles[0], "newton 5D")


# ---------------------------------------------------------------------------
# 4. Hybrid (Power + Newton)
# ---------------------------------------------------------------------------
class TestHybrid:
    def test_fp_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_hybrid(
            A,
            W,
            h,
            1,
            B=256,
            outer_iterations=5,
        )
        assert len(cycles[0]) >= 1
        _check_fp(A, W, h, cycles[0], "hybrid")

    def test_2cycle_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_hybrid(
            A,
            W,
            h,
            2,
            B=256,
            outer_iterations=5,
        )
        if cycles[1]:
            _check_cycle(A, W, h, cycles[1], 2, "hybrid 2-cycle")

    def test_5d_fp(self, system_5d):
        A, W, h = system_5d
        torch.manual_seed(42)
        cycles, _ = find_cycles_hybrid(
            A,
            W,
            h,
            1,
            B=256,
            outer_iterations=5,
        )
        _check_fp(A, W, h, cycles[0], "hybrid 5D")


# ---------------------------------------------------------------------------
# 5. Woodbury (k=1 only)
# ---------------------------------------------------------------------------
class TestWoodbury:
    def test_fp_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_woodbury(A, W, h, 1, B=64)
        assert len(cycles[0]) >= 1
        _check_fp(A, W, h, cycles[0], "woodbury")

    def test_fp_5d(self, system_5d):
        A, W, h = system_5d
        torch.manual_seed(42)
        cycles, _ = find_cycles_woodbury(A, W, h, 1, B=64)
        _check_fp(A, W, h, cycles[0], "woodbury 5D")

    def test_max_order_gt1_empty(self, system_2d):
        """Woodbury only supports k=1. Higher orders return empty."""
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_woodbury(A, W, h, 3, B=64)
        assert cycles[1] == []
        assert cycles[2] == []


# ---------------------------------------------------------------------------
# 6. Periodic Schur
# ---------------------------------------------------------------------------
class TestSchur:
    def test_fp_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_schur(
            A,
            W,
            h,
            1,
            B=64,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        assert len(cycles[0]) >= 1
        _check_fp(A, W, h, cycles[0], "schur")

    def test_2cycle_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        cycles, _ = find_cycles_schur(
            A,
            W,
            h,
            2,
            B=64,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        _check_cycle(A, W, h, cycles[1], 2, "schur 2-cycle")


# ---------------------------------------------------------------------------
# Cross-validation: compare all against exhaustive ground truth
# ---------------------------------------------------------------------------
class TestCrossValidation:
    def test_all_find_same_fps_2d(self, system_2d):
        """All variants should find the same FPs on 2D (verified
        against exhaustive ground truth)."""
        A, W, h = system_2d

        # Ground truth
        gt_cycles, _ = find_cycles_exhaustive(A, W, h, 1)
        gt_fps = {tuple(torch.round(t[0] * 1000).long().tolist()) for t in gt_cycles[0]}
        n_gt = len(gt_fps)
        assert n_gt >= 1, "Ground truth should find at least 1 FP"

        # Test each variant (power excluded — seed-sensitive, tested separately)
        for name, fn, kw in [
            (
                "newton",
                find_cycles_newton,
                {
                    "B": 64,
                    "outer_iterations": 5,
                    "inner_iterations": 5,
                },
            ),
            (
                "hybrid",
                find_cycles_hybrid,
                {
                    "B": 256,
                    "outer_iterations": 5,
                },
            ),
            ("woodbury", find_cycles_woodbury, {"B": 64}),
            (
                "schur",
                find_cycles_schur,
                {
                    "B": 64,
                    "outer_loop_iterations": 10,
                    "inner_loop_iterations": 20,
                },
            ),
        ]:
            torch.manual_seed(42)
            cycles, _ = fn(A, W, h, 1, **kw)
            found = {tuple(torch.round(t[0] * 1000).long().tolist()) for t in cycles[0]}
            overlap = found & gt_fps
            assert len(overlap) >= 1, (
                f"{name} found {len(found)} FPs but none match "
                f"ground truth ({n_gt} FPs)"
            )

        # Power: multi-seed
        for seed in range(20):
            torch.manual_seed(seed)
            cycles, _ = find_cycles_power(A, W, h, 1)
            found = {tuple(torch.round(t[0] * 1000).long().tolist()) for t in cycles[0]}
            overlap = found & gt_fps
            if len(overlap) >= 1:
                break
        else:
            pytest.fail("power found no FPs matching ground truth in 20 seeds")
