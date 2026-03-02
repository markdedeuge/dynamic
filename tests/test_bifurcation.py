"""TDD tests for bifurcation detection utilities (Phase 4 extension).

Tests written FIRST per TDD methodology. Implementation target:
    dynamic.analysis.bifurcation

These test the bifurcation detection logic: stability comparison,
distance metrics, parameter grid construction, and bifurcation
detection along grids and training trajectories.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def simple_2d_params():
    """Base 2D PLRNN parameters for grid tests."""
    A = torch.tensor([[0.5, 0.0], [0.0, -0.3]])
    W = torch.tensor([[0.0, -0.6], [0.5, 0.0]])
    h = torch.tensor([0.37, -0.98])
    return A, W, h


# ---------------------------------------------------------------------------
# compare_stability
# ---------------------------------------------------------------------------
class TestCompareStability:
    """Tests for compare_stability function."""

    def test_same_stability(self):
        """Matching stable dimension counts → returns index list."""
        from dynamic.analysis.bifurcation import compare_stability

        eigvals = np.array([0.5, 0.3])  # 2 stable dims
        eigvals_neighbours = [
            np.array([0.4, 0.6]),  # 2 stable dims → match
            np.array([1.5, 0.3]),  # 1 stable dim → no match
        ]
        result = compare_stability(eigvals, eigvals_neighbours)
        assert result is not None
        assert 0 in result
        assert 1 not in result

    def test_no_match(self):
        """No matching stability → returns None."""
        from dynamic.analysis.bifurcation import compare_stability

        eigvals = np.array([0.5, 0.3])  # 2 stable
        eigvals_neighbours = [
            np.array([1.5, 1.3]),  # 0 stable
            np.array([1.5, 0.3]),  # 1 stable
        ]
        result = compare_stability(eigvals, eigvals_neighbours)
        assert result is None

    def test_multiple_matches(self):
        """Multiple neighbours with same stability → all returned."""
        from dynamic.analysis.bifurcation import compare_stability

        eigvals = np.array([0.5, 1.5])  # 1 stable dim
        eigvals_neighbours = [
            np.array([0.4, 1.2]),  # 1 stable → match
            np.array([0.9, 2.0]),  # 1 stable → match
            np.array([0.1, 0.2]),  # 2 stable → no match
        ]
        result = compare_stability(eigvals, eigvals_neighbours)
        assert result is not None
        assert 0 in result
        assert 1 in result
        assert 2 not in result

    def test_empty_neighbours(self):
        """Empty neighbour list → None."""
        from dynamic.analysis.bifurcation import compare_stability

        eigvals = np.array([0.5, 0.3])
        result = compare_stability(eigvals, [])
        assert result is None

    def test_complex_eigenvalues(self):
        """Complex eigenvalues — uses magnitudes for stability."""
        from dynamic.analysis.bifurcation import compare_stability

        # |0.3+0.4j| = 0.5 < 1, so 1 stable dim (complex conjugate pair)
        eigvals = np.array([0.3 + 0.4j, 0.3 - 0.4j])
        eigvals_neighbours = [
            np.array([0.1 + 0.2j, 0.1 - 0.2j]),  # 2 stable → match
        ]
        result = compare_stability(eigvals, eigvals_neighbours)
        assert result is not None
        assert 0 in result


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------
class TestMinimalStateSpaceDistances:
    """Tests for get_minimal_state_space_distances."""

    def test_identical_cycle(self):
        """Identical cycle → distance 0."""
        from dynamic.analysis.bifurcation import (
            get_minimal_state_space_distances,
        )

        cycle = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        neighbours = [cycle]
        dists = get_minimal_state_space_distances(cycle, neighbours)
        assert len(dists) == 1
        assert dists[0] < 1e-6

    def test_rotated_cycle(self):
        """Cyclic rotation of same cycle → distance 0."""
        from dynamic.analysis.bifurcation import (
            get_minimal_state_space_distances,
        )

        cycle = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        rotated = [torch.tensor([3.0, 4.0]), torch.tensor([1.0, 2.0])]
        dists = get_minimal_state_space_distances(cycle, [rotated])
        assert len(dists) == 1
        assert dists[0] < 1e-6

    def test_different_cycles(self):
        """Clearly different cycles → positive distance."""
        from dynamic.analysis.bifurcation import (
            get_minimal_state_space_distances,
        )

        cycle = [torch.tensor([0.0, 0.0])]
        far_cycle = [torch.tensor([10.0, 10.0])]
        dists = get_minimal_state_space_distances(cycle, [far_cycle])
        assert dists[0] > 1.0

    def test_multiple_neighbours(self):
        """Returns one distance per neighbour."""
        from dynamic.analysis.bifurcation import (
            get_minimal_state_space_distances,
        )

        cycle = [torch.tensor([1.0, 0.0])]
        neighbours = [
            [torch.tensor([1.1, 0.0])],
            [torch.tensor([5.0, 5.0])],
        ]
        dists = get_minimal_state_space_distances(cycle, neighbours)
        assert len(dists) == 2
        assert dists[0] < dists[1]


class TestMinimalEigenvalueDistances:
    """Tests for get_minimal_eigenvalue_distances."""

    def test_identical_eigenvalues(self):
        """Same eigenvalues → distance 0."""
        from dynamic.analysis.bifurcation import (
            get_minimal_eigenvalue_distances,
        )

        eigvals = np.array([0.5, -0.3])
        neighbours = [np.array([0.5, -0.3])]
        dists = get_minimal_eigenvalue_distances(eigvals, neighbours)
        assert len(dists) == 1
        assert dists[0] < 1e-6

    def test_different_eigenvalues(self):
        """Different eigenvalues → positive distance."""
        from dynamic.analysis.bifurcation import (
            get_minimal_eigenvalue_distances,
        )

        eigvals = np.array([0.5, -0.3])
        neighbours = [np.array([2.0, 3.0])]
        dists = get_minimal_eigenvalue_distances(eigvals, neighbours)
        assert dists[0] > 1.0


class TestCombinedDistance:
    """Tests for get_combined_state_space_eigenvalue_distance."""

    def test_combined_is_sum(self):
        """Combined distance = state_space + eigenvalue distances."""
        from dynamic.analysis.bifurcation import (
            get_combined_state_space_eigenvalue_distance,
            get_minimal_eigenvalue_distances,
            get_minimal_state_space_distances,
        )

        cycle = [torch.tensor([1.0, 2.0])]
        neighbours = [[torch.tensor([1.1, 2.1])]]
        eigval = np.array([0.5])
        eigval_neighbours = [np.array([0.6])]

        combined = get_combined_state_space_eigenvalue_distance(
            cycle, neighbours, eigval, eigval_neighbours,
        )
        ss = get_minimal_state_space_distances(cycle, neighbours)
        ev = get_minimal_eigenvalue_distances(eigval, eigval_neighbours)
        np.testing.assert_allclose(combined, [ss[0] + ev[0]], atol=1e-6)


# ---------------------------------------------------------------------------
# Parameter grid construction
# ---------------------------------------------------------------------------
class TestCreateGridData:
    """Tests for create_grid_data."""

    def test_grid_size(self, simple_2d_params):
        """Grid produces len(p1) * len(p2) entries."""
        from dynamic.analysis.bifurcation import create_grid_data

        A, W, h = simple_2d_params
        p1 = [0.0, 0.1, 0.2]
        p2 = [0.0, 0.5]
        grid = create_grid_data(A, W, h, "a11", "a22", p1, p2)
        assert len(grid) == 6  # 3 * 2

    def test_grid_entry_structure(self, simple_2d_params):
        """Each entry is (A_mod, W_mod, h_mod) tuple."""
        from dynamic.analysis.bifurcation import create_grid_data

        A, W, h = simple_2d_params
        p1 = [0.1]
        p2 = [0.2]
        grid = create_grid_data(A, W, h, "a11", "w12", p1, p2)
        assert len(grid) == 1
        entry = grid[0]
        assert len(entry) == 3  # (A, W, h)
        assert entry[0].shape == (2, 2)  # A
        assert entry[1].shape == (2, 2)  # W
        assert entry[2].shape == (2,)    # h

    def test_a11_modification(self, simple_2d_params):
        """Param 'a11' modifies A[0,0]."""
        from dynamic.analysis.bifurcation import create_grid_data

        A, W, h = simple_2d_params
        delta = 0.5
        grid = create_grid_data(A, W, h, "a11", "a11", [delta], [0.0])
        A_mod = grid[0][0]
        assert abs(A_mod[0, 0].item() - (A[0, 0].item() + delta)) < 1e-6

    def test_w21_modification(self, simple_2d_params):
        """Param 'w21' modifies W[1,0]."""
        from dynamic.analysis.bifurcation import create_grid_data

        A, W, h = simple_2d_params
        delta = 0.3
        grid = create_grid_data(A, W, h, "w21", "a11", [delta], [0.0])
        W_mod = grid[0][1]
        assert abs(W_mod[1, 0].item() - (W[1, 0].item() + delta)) < 1e-6

    def test_h1_modification(self, simple_2d_params):
        """Param 'h1' modifies h[0]."""
        from dynamic.analysis.bifurcation import create_grid_data

        A, W, h = simple_2d_params
        delta = -0.2
        grid = create_grid_data(A, W, h, "h1", "a11", [delta], [0.0])
        h_mod = grid[0][2]
        assert abs(h_mod[0].item() - (h[0].item() + delta)) < 1e-6


# ---------------------------------------------------------------------------
# Bifurcation detection along trajectory
# ---------------------------------------------------------------------------
class TestFindBifurcationsTrajectory:
    """Tests for find_bifurcations_trajectory."""

    def test_no_bifurcation(self):
        """Constant dynamics → no bifurcation epochs."""
        from dynamic.analysis.bifurcation import (
            find_bifurcations_trajectory,
        )

        # Two consecutive snapshots with same cycle structure
        cycles_list = [
            [[torch.tensor([1.0, 2.0])]],  # order 1: 1 FP
            [[torch.tensor([1.1, 2.1])]],  # order 1: 1 FP (shifted slightly)
        ]
        eigvals_list = [
            [np.array([0.5, 0.3])],
            [np.array([0.5, 0.3])],
        ]
        epochs = find_bifurcations_trajectory(
            cycles_list, eigvals_list, list(range(len(cycles_list)))
        )
        assert len(epochs) == 0

    def test_number_change_bifurcation(self):
        """Cycle count changes → bifurcation detected."""
        from dynamic.analysis.bifurcation import (
            find_bifurcations_trajectory,
        )

        cycles_list = [
            [[torch.tensor([1.0, 2.0])]],
            [[torch.tensor([1.0, 2.0])], [torch.tensor([3.0, 4.0])]],
        ]
        eigvals_list = [
            [np.array([0.5, 0.3])],
            [np.array([0.5, 0.3]), np.array([0.4, 0.2])],
        ]
        epochs = find_bifurcations_trajectory(
            cycles_list, eigvals_list, list(range(len(cycles_list)))
        )
        assert len(epochs) >= 1

    def test_stability_change_bifurcation(self):
        """Stability changes → bifurcation detected."""
        from dynamic.analysis.bifurcation import (
            find_bifurcations_trajectory,
        )

        cycles_list = [
            [[torch.tensor([1.0, 2.0])]],
            [[torch.tensor([1.0, 2.0])]],
        ]
        eigvals_list = [
            [np.array([0.5, 0.3])],   # 2 stable dims
            [np.array([1.5, 1.3])],    # 0 stable dims
        ]
        epochs = find_bifurcations_trajectory(
            cycles_list, eigvals_list, list(range(len(cycles_list)))
        )
        assert len(epochs) >= 1

    def test_returns_epoch_indices(self):
        """Bifurcation epochs are returned as model indices."""
        from dynamic.analysis.bifurcation import (
            find_bifurcations_trajectory,
        )

        model_numbers = [10, 20, 30]
        cycles_list = [
            [[torch.tensor([1.0, 2.0])]],
            [[torch.tensor([1.0, 2.0])]],
            [],  # FP disappears
        ]
        eigvals_list = [
            [np.array([0.5, 0.3])],
            [np.array([0.5, 0.3])],
            [],
        ]
        epochs = find_bifurcations_trajectory(
            cycles_list, eigvals_list, model_numbers
        )
        assert len(epochs) >= 1
        # At least one epoch should reference model 20
        assert any(e == 20 for e in epochs)


# ---------------------------------------------------------------------------
# Bifurcation detection on parameter grid
# ---------------------------------------------------------------------------
class TestFindBifurcationsGrid:
    """Tests for find_bifurcations_parameter_grid."""

    def test_uniform_grid_no_bifurcation(self):
        """All grid points same structure → no bifurcations."""
        from dynamic.analysis.bifurcation import (
            find_bifurcations_parameter_grid,
        )

        n = 3
        cycles_grid = []
        eigvals_grid = []
        coords_grid = []
        for i in range(n):
            for j in range(n):
                cycles_grid.append(
                    [[torch.tensor([1.0, 2.0])]]
                )
                eigvals_grid.append(
                    [np.array([0.5, 0.3])]
                )
                coords_grid.append((float(i), float(j)))

        bifurcations = find_bifurcations_parameter_grid(
            cycles_grid, eigvals_grid, coords_grid, n
        )
        assert len(bifurcations) == 0

    def test_detects_count_change(self):
        """Grid with cycle count change → bifurcation detected."""
        from dynamic.analysis.bifurcation import (
            find_bifurcations_parameter_grid,
        )

        n = 2
        # 2x2 grid: top-left has 1 FP, top-right has 2 FPs
        cycles_grid = [
            [[torch.tensor([1.0, 2.0])]],           # (0,0)
            [[torch.tensor([1.0, 2.0])],
             [torch.tensor([3.0, 4.0])]],            # (0,1)
            [[torch.tensor([1.0, 2.0])]],            # (1,0)
            [[torch.tensor([1.0, 2.0])]],            # (1,1)
        ]
        eigvals_grid = [
            [np.array([0.5, 0.3])],
            [np.array([0.5, 0.3]), np.array([0.4, 0.2])],
            [np.array([0.5, 0.3])],
            [np.array([0.5, 0.3])],
        ]
        coords_grid = [(0, 0), (0, 1), (1, 0), (1, 1)]

        bifurcations = find_bifurcations_parameter_grid(
            cycles_grid, eigvals_grid, coords_grid, n
        )
        assert len(bifurcations) >= 1

    def test_bifurcation_has_coordinates(self):
        """Each bifurcation entry contains grid coordinates."""
        from dynamic.analysis.bifurcation import (
            find_bifurcations_parameter_grid,
        )

        n = 2
        cycles_grid = [
            [[torch.tensor([1.0, 2.0])]],
            [],  # No cycles at (0,1) — count change
            [[torch.tensor([1.0, 2.0])]],
            [[torch.tensor([1.0, 2.0])]],
        ]
        eigvals_grid = [
            [np.array([0.5, 0.3])],
            [],
            [np.array([0.5, 0.3])],
            [np.array([0.5, 0.3])],
        ]
        coords_grid = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

        bifurcations = find_bifurcations_parameter_grid(
            cycles_grid, eigvals_grid, coords_grid, n
        )
        assert len(bifurcations) >= 1
        for bif in bifurcations:
            assert "coord_1" in bif
            assert "coord_2" in bif
            assert "type" in bif
