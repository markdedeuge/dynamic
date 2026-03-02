"""TDD tests for subregion utilities (Phase 4, Round 1).

Tests written FIRST per TDD methodology. Implementation target:
    dynamic.analysis.subregions
"""

import numpy as np
import torch


class TestGetD:
    """Tests for the standalone get_D function."""

    def test_all_positive(self):
        """All-positive z → identity diagonal."""
        from dynamic.analysis.subregions import get_D

        z = torch.tensor([1.0, 2.0, 3.0])
        D = get_D(z)
        expected = torch.eye(3)
        assert torch.allclose(D, expected)

    def test_mixed_signs(self):
        """Mixed signs → correct binary diagonal."""
        from dynamic.analysis.subregions import get_D

        z = torch.tensor([1.0, -2.0, 0.5, -0.1])
        D = get_D(z)
        expected = torch.diag(torch.tensor([1.0, 0.0, 1.0, 0.0]))
        assert torch.allclose(D, expected)

    def test_zero_boundary(self):
        """z_i = 0 maps to d_i = 0 (ReLU convention)."""
        from dynamic.analysis.subregions import get_D

        z = torch.tensor([0.0, 1.0, -1.0, 0.0])
        D = get_D(z)
        diag = torch.diag(D)
        assert diag[0] == 0.0
        assert diag[1] == 1.0
        assert diag[2] == 0.0
        assert diag[3] == 0.0

    def test_all_negative(self):
        """All-negative z → zero diagonal."""
        from dynamic.analysis.subregions import get_D

        z = torch.tensor([-1.0, -2.0, -3.0])
        D = get_D(z)
        expected = torch.zeros(3, 3)
        assert torch.allclose(D, expected)

    def test_output_shape(self):
        """Output is (M, M) for input of shape (M,)."""
        from dynamic.analysis.subregions import get_D

        z = torch.randn(7)
        D = get_D(z)
        assert D.shape == (7, 7)


class TestGetRegionId:
    """Tests for the get_region_id function."""

    def test_type(self):
        """Returns a hashable tuple of ints."""
        from dynamic.analysis.subregions import get_region_id

        z = torch.randn(5)
        rid = get_region_id(z)
        assert isinstance(rid, tuple)
        assert all(isinstance(x, int) for x in rid)
        assert all(x in (0, 1) for x in rid)

    def test_deterministic(self):
        """Same z always gives same region ID."""
        from dynamic.analysis.subregions import get_region_id

        z = torch.tensor([1.0, -1.0, 0.5, -0.5])
        assert get_region_id(z) == get_region_id(z)

    def test_all_positive(self):
        """All-positive → all-ones tuple."""
        from dynamic.analysis.subregions import get_region_id

        z = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert get_region_id(z) == (1, 1, 1, 1)

    def test_all_negative(self):
        """All-negative → all-zeros tuple."""
        from dynamic.analysis.subregions import get_region_id

        z = torch.tensor([-1.0, -2.0, -3.0])
        assert get_region_id(z) == (0, 0, 0)

    def test_length_matches_dim(self):
        """Length of tuple matches input dimensionality."""
        from dynamic.analysis.subregions import get_region_id

        z = torch.randn(10)
        assert len(get_region_id(z)) == 10

    def test_hashable(self):
        """Can be used as a dict key or set element."""
        from dynamic.analysis.subregions import get_region_id

        z = torch.randn(4)
        rid = get_region_id(z)
        d = {rid: 42}
        assert d[rid] == 42


class TestGetNeighbors:
    """Tests for the get_neighbors function."""

    def test_count(self):
        """M-dim point has exactly M neighbors (one per bit flip)."""
        from dynamic.analysis.subregions import get_neighbors

        region = (1, 0, 1, 0)
        neighbors = get_neighbors(region)
        assert len(neighbors) == 4

    def test_hamming_distance(self):
        """Each neighbor differs by exactly 1 bit."""
        from dynamic.analysis.subregions import get_neighbors

        region = (1, 0, 1)
        neighbors = get_neighbors(region)
        for n in neighbors:
            diff = sum(a != b for a, b in zip(region, n))
            assert diff == 1

    def test_all_unique(self):
        """All neighbors are distinct."""
        from dynamic.analysis.subregions import get_neighbors

        region = (1, 1, 0, 0, 1)
        neighbors = get_neighbors(region)
        assert len(set(neighbors)) == len(neighbors)

    def test_neighbor_type(self):
        """Each neighbor is a tuple of ints."""
        from dynamic.analysis.subregions import get_neighbors

        region = (0, 1)
        neighbors = get_neighbors(region)
        for n in neighbors:
            assert isinstance(n, tuple)
            assert all(isinstance(x, int) for x in n)
            assert all(x in (0, 1) for x in n)


class TestGetJacobianInRegion:
    """Tests for Jacobian computation in a given subregion."""

    def test_plrnn_formula(self):
        """PLRNN Jacobian: A + W·D matches manual computation."""
        from dynamic.analysis.subregions import get_jacobian_in_region

        A = torch.tensor([[0.5, 0.0], [0.0, -0.3]])
        W = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        D = torch.tensor([[1.0, 0.0], [0.0, 0.0]])  # z1 > 0, z2 < 0

        J = get_jacobian_in_region(A, W, D)
        expected = A + W @ D
        assert torch.allclose(J, expected, atol=1e-6)

    def test_identity_D(self):
        """D = I → Jacobian is A + W."""
        from dynamic.analysis.subregions import get_jacobian_in_region

        A = torch.randn(3, 3)
        W = torch.randn(3, 3)
        D = torch.eye(3)
        J = get_jacobian_in_region(A, W, D)
        expected = A + W
        assert torch.allclose(J, expected, atol=1e-6)

    def test_zero_D(self):
        """D = 0 → Jacobian is just A."""
        from dynamic.analysis.subregions import get_jacobian_in_region

        A = torch.randn(3, 3)
        W = torch.randn(3, 3)
        D = torch.zeros(3, 3)
        J = get_jacobian_in_region(A, W, D)
        assert torch.allclose(J, A, atol=1e-6)

    def test_output_shape(self):
        """Output shape is (M, M)."""
        from dynamic.analysis.subregions import get_jacobian_in_region

        A = torch.randn(5, 5)
        W = torch.randn(5, 5)
        D = torch.eye(5)
        J = get_jacobian_in_region(A, W, D)
        assert J.shape == (5, 5)


class TestClassifyPoint:
    """Tests for eigenvalue-based stability classification."""

    def test_stable(self):
        """All |λ| < 1 → 'stable'."""
        from dynamic.analysis.subregions import classify_point

        eigenvalues = np.array([0.5 + 0.1j, 0.3 - 0.2j, -0.4])
        assert classify_point(eigenvalues) == "stable"

    def test_unstable(self):
        """All |λ| > 1 → 'unstable'."""
        from dynamic.analysis.subregions import classify_point

        eigenvalues = np.array([1.5, -2.0, 1.1 + 0.5j])
        assert classify_point(eigenvalues) == "unstable"

    def test_saddle(self):
        """Mixed magnitudes → 'saddle'."""
        from dynamic.analysis.subregions import classify_point

        eigenvalues = np.array([0.5, 1.5, -0.3])
        assert classify_point(eigenvalues) == "saddle"

    def test_boundary_stable(self):
        """Eigenvalue with |λ| = 1 exactly is NOT stable (use strict <)."""
        from dynamic.analysis.subregions import classify_point

        eigenvalues = np.array([1.0, 0.5])
        # |1.0| is not < 1. Since not all > 1, and not all < 1 → saddle
        assert classify_point(eigenvalues) == "saddle"

    def test_real_eigenvalues(self):
        """Works with purely real eigenvalues."""
        from dynamic.analysis.subregions import classify_point

        eigenvalues = np.array([0.9, 0.8, 0.7])
        assert classify_point(eigenvalues) == "stable"

    def test_complex_conjugates(self):
        """Complex conjugate pair with |λ| < 1 → stable."""
        from dynamic.analysis.subregions import classify_point

        eigenvalues = np.array([0.3 + 0.4j, 0.3 - 0.4j])
        # |0.3 + 0.4j| = 0.5 < 1
        assert classify_point(eigenvalues) == "stable"
