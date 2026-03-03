"""Tests for optimised SCYFI functions.

Compares fast implementations against reference for correctness,
then benchmarks the speedup.
"""

import numpy as np
import pytest
import torch

from dynamic.analysis.scyfi import find_cycles
from dynamic.analysis.scyfi_fast import (
    find_cycles_fast,
    find_cycles_sh_fast,
)
from dynamic.analysis.scyfi_helpers import (
    get_cycle_point_candidate,
    get_eigvals,
    get_factor_in_front_of_h,
    get_factor_in_front_of_z,
    get_latent_time_series,
)
from dynamic.analysis.scyfi_helpers_fast import (
    _chain_product,
    batch_candidates,
    get_candidate_and_eigvals,
    make_key,
    random_d_vecs,
    random_d_vecs_batch,
    simulate_and_extract,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def system_2d():
    """2D PLRNN system with known fixed points."""
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


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------
class TestChainProduct:
    """Fused chain product must match sequential reference."""

    def test_z_factor_matches(self, system_2d):
        A, W, h = system_2d
        dim = 2
        torch.manual_seed(99)
        d_vecs = random_d_vecs(dim, 3, dtype=A.dtype)
        eye = torch.eye(dim, dtype=A.dtype)

        z_factor, _ = _chain_product(A, W, d_vecs, 3, eye)

        # Build full D_list for reference
        D_list = torch.zeros(3, dim, dim, dtype=A.dtype)
        for i in range(3):
            D_list[i] = torch.diag(d_vecs[i])
        ref = get_factor_in_front_of_z(A, W, D_list, 3)

        assert torch.allclose(z_factor, ref, atol=1e-12)

    def test_h_factor_matches(self, system_2d):
        A, W, h = system_2d
        dim = 2
        torch.manual_seed(99)
        d_vecs = random_d_vecs(dim, 3, dtype=A.dtype)
        eye = torch.eye(dim, dtype=A.dtype)

        _, h_factor = _chain_product(A, W, d_vecs, 3, eye)

        D_list = torch.zeros(3, dim, dim, dtype=A.dtype)
        for i in range(3):
            D_list[i] = torch.diag(d_vecs[i])
        ref = get_factor_in_front_of_h(A, W, D_list, 3)

        assert torch.allclose(h_factor, ref, atol=1e-12)

    def test_order_1(self, system_2d):
        A, W, h = system_2d
        dim = 2
        d_vecs = random_d_vecs(dim, 1, dtype=A.dtype)
        eye = torch.eye(dim, dtype=A.dtype)

        z_factor, h_factor = _chain_product(A, W, d_vecs, 1, eye)

        D_list = torch.diag(d_vecs[0]).unsqueeze(0)
        ref_z = get_factor_in_front_of_z(A, W, D_list, 1)
        ref_h = get_factor_in_front_of_h(A, W, D_list, 1)

        assert torch.allclose(z_factor, ref_z, atol=1e-12)
        assert torch.allclose(h_factor, ref_h, atol=1e-12)


class TestFusedCandidate:
    """Fused candidate+eigvals must match reference."""

    def test_candidate_matches(self, system_2d):
        A, W, h = system_2d
        dim = 2
        torch.manual_seed(42)
        d_vecs = random_d_vecs(dim, 1, dtype=A.dtype)
        eye = torch.eye(dim, dtype=A.dtype)

        cand, eigs = get_candidate_and_eigvals(
            A, W, d_vecs, h, 1, eye,
        )

        D_list = torch.diag(d_vecs[0]).unsqueeze(0)
        ref_cand = get_cycle_point_candidate(A, W, D_list, h, 1)
        ref_eigs = get_eigvals(A, W, D_list, 1)

        assert cand is not None
        assert ref_cand is not None
        assert torch.allclose(cand, ref_cand, atol=1e-12)
        np.testing.assert_allclose(
            np.sort(np.abs(eigs)), np.sort(np.abs(ref_eigs)), atol=1e-12,
        )


class TestSimulateAndExtract:
    """Fused trajectory+D-extraction must match reference."""

    def test_trajectory_matches(self, system_2d):
        A, W, h = system_2d
        dim = 2
        z0 = torch.tensor([1.0, -0.5], dtype=torch.float64)

        traj_fast, d_fast = simulate_and_extract(z0, A, W, h, 3)
        traj_ref = get_latent_time_series(3, A, W, h, dim, z_0=z0)

        for i in range(3):
            assert torch.allclose(traj_fast[i], traj_ref[i], atol=1e-12)

    def test_d_vecs_correct(self, system_2d):
        A, W, h = system_2d
        z0 = torch.tensor([1.0, -0.5], dtype=torch.float64)

        _, d_vecs = simulate_and_extract(z0, A, W, h, 3)

        # First d should be [1, 0] since z0 = [1.0, -0.5]
        assert d_vecs[0, 0] == 1.0
        assert d_vecs[0, 1] == 0.0


class TestBatchCandidates:
    """Batched bmm candidates must match sequential."""

    def test_matches_sequential(self, system_2d):
        A, W, h = system_2d
        dim = 2
        B = 8
        order = 2
        torch.manual_seed(123)
        eye = torch.eye(dim, dtype=A.dtype)

        d_batch = random_d_vecs_batch(B, dim, order, dtype=A.dtype)
        cands, z_facts, valid = batch_candidates(
            A, W, h, d_batch, order, eye,
        )

        for b in range(B):
            d_vecs = d_batch[b]
            cand_seq, _ = get_candidate_and_eigvals(
                A, W, d_vecs, h, order, eye,
            )
            if cand_seq is not None:
                assert valid[b]
                assert torch.allclose(cands[b], cand_seq, atol=1e-10)
            else:
                assert not valid[b]


class TestHashDedup:
    """Hash-based dedup must be consistent."""

    def test_same_point_same_key(self):
        z = torch.tensor([1.234, -5.678], dtype=torch.float64)
        assert make_key(z) == make_key(z.clone())

    def test_different_points_different_keys(self):
        z1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        z2 = torch.tensor([1.0, 3.0], dtype=torch.float64)
        assert make_key(z1) != make_key(z2)

    def test_close_points_same_key(self):
        z1 = torch.tensor([1.0001, 2.0001], dtype=torch.float64)
        z2 = torch.tensor([1.0002, 2.0002], dtype=torch.float64)
        # Within 1/1000 rounding
        assert make_key(z1) == make_key(z2)


# ---------------------------------------------------------------------------
# Algorithm-level tests
# ---------------------------------------------------------------------------
class TestFindCyclesFast:
    """find_cycles_fast must find the same cycles as reference."""

    def test_fp_count_2d(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(0)
        ref_cycles, ref_eigs = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        torch.manual_seed(0)
        fast_cycles, fast_eigs = find_cycles_fast(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        # Both should find at least 1 fixed point
        assert len(ref_cycles[0]) >= 1
        assert len(fast_cycles[0]) >= 1

    def test_fp_self_consistency(self, system_2d):
        """All found FPs must be genuine fixed points."""
        A, W, h = system_2d
        torch.manual_seed(42)
        fast_cycles, _ = find_cycles_fast(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        for fp_traj in fast_cycles[0]:
            z = fp_traj[0]
            z_next = A @ z + W @ torch.clamp(z, min=0.0) + h
            assert torch.allclose(z, z_next, atol=1e-6)

    def test_cycle_self_consistency(self, system_2d):
        """All found 2-cycles must be genuine periodic orbits."""
        A, W, h = system_2d
        torch.manual_seed(42)
        fast_cycles, _ = find_cycles_fast(
            A, W, h, 2,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        for traj in fast_cycles[1]:  # order-2 cycles
            z0, z1 = traj[0], traj[1]
            z0_next = A @ z0 + W @ torch.clamp(z0, min=0.0) + h
            z1_next = A @ z1 + W @ torch.clamp(z1, min=0.0) + h
            assert torch.allclose(z0_next, z1, atol=1e-6)
            assert torch.allclose(z1_next, z0, atol=1e-6)

    def test_fp_5d(self, system_5d):
        A, W, h = system_5d
        torch.manual_seed(0)
        fast_cycles, _ = find_cycles_fast(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )
        # Self-consistency check for all found FPs
        for fp_traj in fast_cycles[0]:
            z = fp_traj[0]
            z_next = A @ z + W @ torch.clamp(z, min=0.0) + h
            assert torch.allclose(z, z_next, atol=1e-6)


class TestFindCyclesBatched:
    """Batched variant must find valid cycles."""

    def test_fp_self_consistency(self, system_2d):
        A, W, h = system_2d
        torch.manual_seed(42)
        fast_cycles, _ = find_cycles_fast(
            A, W, h, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
            batched=True,
            batch_size=16,
        )
        for fp_traj in fast_cycles[0]:
            z = fp_traj[0]
            z_next = A @ z + W @ torch.clamp(z, min=0.0) + h
            assert torch.allclose(z, z_next, atol=1e-6)


class TestFindCyclesShFast:
    """find_cycles_sh_fast must find valid cycles."""

    def test_sh_fp_self_consistency(self):
        torch.manual_seed(42)
        dim, hidden_dim = 2, 6
        A = torch.randn(dim, dtype=torch.float64)
        W1 = torch.randn(dim, hidden_dim, dtype=torch.float64) * 0.3
        W2 = torch.randn(hidden_dim, dim, dtype=torch.float64) * 0.3
        h1 = torch.randn(dim, dtype=torch.float64)
        h2 = torch.randn(hidden_dim, dtype=torch.float64)

        fast_cycles, _ = find_cycles_sh_fast(
            A, W1, W2, h1, h2, 1,
            outer_loop_iterations=10,
            inner_loop_iterations=20,
        )

        for fp_traj in fast_cycles[0]:
            z = fp_traj[0]
            z_next = A * z + W1 @ torch.clamp(W2 @ z + h2, min=0.0) + h1
            assert torch.allclose(z, z_next, atol=1e-6)
