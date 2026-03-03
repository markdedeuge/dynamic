"""Tests for SCYFI solve utilities."""

import pytest
import torch

from dynamic.analysis.scyfi_solve import (
    _solve_2x2,
    _solve_3x3,
    auto_batch_size,
    batched_solve,
    build_awd_table,
    d_vecs_to_indices,
    fast_key,
)


class TestSolve2x2:
    """Closed-form 2×2 batched solve."""

    def test_matches_linalg_solve(self):
        torch.manual_seed(42)
        B = 32
        A = torch.randn(B, 2, 2, dtype=torch.float64)
        # Make well-conditioned
        A += 3.0 * torch.eye(2, dtype=torch.float64).unsqueeze(0)
        b = torch.randn(B, 2, 1, dtype=torch.float64)

        expected = torch.linalg.solve(A, b)
        result = _solve_2x2(A, b)

        assert torch.allclose(result, expected, atol=1e-10), (
            f"Max diff: {(result - expected).abs().max():.2e}"
        )

    def test_singular_returns_nan(self):
        A = torch.tensor([[[1.0, 2.0], [2.0, 4.0]]], dtype=torch.float64)
        b = torch.tensor([[[1.0], [1.0]]], dtype=torch.float64)
        result = _solve_2x2(A, b)
        assert torch.isnan(result).all()

    def test_identity(self):
        B = 8
        A = torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1)
        b = torch.randn(B, 2, 1, dtype=torch.float64)
        result = _solve_2x2(A.clone(), b)
        assert torch.allclose(result, b, atol=1e-12)


class TestSolve3x3:
    """Closed-form 3×3 Cramer's rule solve."""

    def test_matches_linalg_solve(self):
        torch.manual_seed(42)
        B = 32
        A = torch.randn(B, 3, 3, dtype=torch.float64)
        A += 3.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0)
        b = torch.randn(B, 3, 1, dtype=torch.float64)

        expected = torch.linalg.solve(A, b)
        result = _solve_3x3(A, b)

        assert torch.allclose(result, expected, atol=1e-10), (
            f"Max diff: {(result - expected).abs().max():.2e}"
        )

    def test_singular_returns_nan(self):
        A = torch.zeros(1, 3, 3, dtype=torch.float64)
        A[0, 0, 0] = 1.0  # rank 1
        b = torch.ones(1, 3, 1, dtype=torch.float64)
        result = _solve_3x3(A, b)
        assert torch.isnan(result).all()


class TestBatchedSolve:
    """Dispatch to closed-form or linalg.solve."""

    @pytest.mark.parametrize("dim", [2, 3, 5, 10])
    def test_matches_linalg_solve(self, dim):
        torch.manual_seed(42)
        B = 16
        A = torch.randn(B, dim, dim, dtype=torch.float64)
        A += 3.0 * torch.eye(dim, dtype=torch.float64).unsqueeze(0)
        b = torch.randn(B, dim, 1, dtype=torch.float64)

        expected = torch.linalg.solve(A, b)
        result = batched_solve(A, b)

        assert torch.allclose(result, expected, atol=1e-10), (
            f"dim={dim}: Max diff: {(result - expected).abs().max():.2e}"
        )


class TestAWDTable:
    """AWD lookup table and index conversion."""

    def test_table_matches_manual(self):
        torch.manual_seed(42)
        dim = 3
        A = torch.diag(torch.randn(dim, dtype=torch.float64))
        W = torch.randn(dim, dim, dtype=torch.float64)
        table = build_awd_table(A, W)

        assert table.shape == (8, 3, 3)  # 2^3 = 8

        # Check specific patterns
        d = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        expected = A + W * d
        idx = d_vecs_to_indices(d)
        assert torch.allclose(table[idx], expected)

    def test_all_patterns_dim2(self):
        dim = 2
        A = torch.tensor(
            [[0.5, 0.0], [0.0, -0.3]],
            dtype=torch.float64,
        )
        W = torch.tensor(
            [[0.0, -0.6], [0.5, 0.0]],
            dtype=torch.float64,
        )
        table = build_awd_table(A, W)

        for i in range(4):
            d = torch.tensor(
                [(i >> b) & 1 for b in range(dim)],
                dtype=torch.float64,
            )
            expected = A + W * d
            assert torch.allclose(table[i], expected), f"Mismatch at index {i}, d={d}"

    def test_indices_roundtrip(self):
        torch.manual_seed(42)
        B = 32
        dim = 5
        d_vecs = torch.randint(0, 2, (B, dim), dtype=torch.float64)
        indices = d_vecs_to_indices(d_vecs)

        # Reconstruct from indices
        for i in range(B):
            idx = indices[i].item()
            for bit in range(dim):
                assert d_vecs[i, bit].item() == ((idx >> bit) & 1)

    def test_batched_lookup(self):
        """Table lookup matches broadcast for batched D-vecs."""
        torch.manual_seed(42)
        dim = 4
        A = torch.diag(torch.randn(dim, dtype=torch.float64))
        W = torch.randn(dim, dim, dtype=torch.float64)
        table = build_awd_table(A, W)

        B = 16
        d_vecs = torch.randint(0, 2, (B, dim), dtype=torch.float64)
        indices = d_vecs_to_indices(d_vecs)

        # Lookup
        AWDs_table = table[indices]  # (B, dim, dim)

        # Manual
        AWDs_manual = A.unsqueeze(0) + W.unsqueeze(0) * d_vecs.unsqueeze(1)

        assert torch.allclose(AWDs_table, AWDs_manual), (
            f"Max diff: {(AWDs_table - AWDs_manual).abs().max():.2e}"
        )


class TestAutoBatchSize:
    """Adaptive batch size heuristic."""

    def test_small_dim(self):
        assert auto_batch_size(2) == 16
        assert auto_batch_size(3) == 16

    def test_medium_dim(self):
        assert auto_batch_size(5) == 32
        assert auto_batch_size(10) == 32

    def test_large_dim(self):
        assert auto_batch_size(20) == 64
        assert auto_batch_size(50) == 64


class TestFastKey:
    """Fast hash key generation."""

    def test_deterministic(self):
        z = torch.tensor([0.123, -0.456, 0.789], dtype=torch.float64)
        k1 = fast_key(z)
        k2 = fast_key(z)
        assert k1 == k2

    def test_different_for_different_points(self):
        z1 = torch.tensor([0.1, 0.2], dtype=torch.float64)
        z2 = torch.tensor([0.1, 0.3], dtype=torch.float64)
        assert fast_key(z1) != fast_key(z2)

    def test_hashable(self):
        """Must work as dict/set key."""
        z1 = torch.tensor([0.1, 0.2], dtype=torch.float64)
        z2 = torch.tensor([0.3, 0.4], dtype=torch.float64)
        s = {fast_key(z1), fast_key(z2)}
        assert len(s) == 2
        assert fast_key(z1) in s

    def test_quantisation_tolerance(self):
        """Nearby points within tolerance should hash the same."""
        z1 = torch.tensor([0.1005, 0.2005], dtype=torch.float64)
        z2 = torch.tensor([0.1004, 0.2004], dtype=torch.float64)
        # At scale=1000: both round to [100, 200]
        assert fast_key(z1) == fast_key(z2)
