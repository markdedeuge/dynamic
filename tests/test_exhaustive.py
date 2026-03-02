"""TDD tests for exhaustive search verification (Phase 4, Round 4).

Tests verify the exhaustive brute-force search agrees with the
heuristic SCYFI algorithm on small 2D systems.

Implementation target:
    dynamic.analysis.exhaustive_search
"""

import pytest
import torch


@pytest.fixture()
def system_2d_simple():
    """Simple 2D PLRNN from Julia exhaustive test."""
    A = torch.tensor([[0.4688040021749482, 0.0], [0.0, -0.6124266402970175]])
    W = torch.tensor([[0.0, 1.7253921307363493], [-0.22529820688589786, 0.0]])
    h = torch.tensor([1.6024054836526365, 0.6123993108987379])
    return A, W, h


class TestExhaustiveSearch:
    """Tests for the brute-force exhaustive search."""

    def test_finds_all_fps_2d(self, system_2d_simple):
        """Exhaustive search finds fixed points in a 2D system."""
        from dynamic.analysis.exhaustive_search import main_exhaustive

        A, W, h = system_2d_simple
        result = main_exhaustive(A, W, h, 1)
        fps = result[0][0]
        # Should find at least 1 FP
        assert len(fps) >= 1

    def test_exhaustive_vs_heuristic_agreement(self, system_2d_simple):
        """Exhaustive and heuristic find the same cycles."""
        from dynamic.analysis.exhaustive_search import main_exhaustive
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_simple
        max_order = 4

        ex_result = main_exhaustive(A, W, h, max_order)
        heur_result = find_cycles(
            A, W, h, max_order,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )

        # For each order, compare counts
        for order_idx in range(max_order):
            ex_count = len(ex_result[0][order_idx])
            heur_count = len(heur_result[0][order_idx])
            assert ex_count == heur_count, (
                f"Order {order_idx + 1}: exhaustive={ex_count}, "
                f"heuristic={heur_count}"
            )

    def test_exhaustive_no_extra_cycles(self, system_2d_simple):
        """Exhaustive doesn't find cycles missed by heuristic (order ≤ 2)."""
        from dynamic.analysis.exhaustive_search import main_exhaustive
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_simple
        max_order = 2

        ex_result = main_exhaustive(A, W, h, max_order)
        heur_result = find_cycles(
            A, W, h, max_order,
            outer_loop_iterations=20, inner_loop_iterations=50,
        )

        for order_idx in range(max_order):
            ex_count = len(ex_result[0][order_idx])
            heur_count = len(heur_result[0][order_idx])
            assert ex_count <= heur_count or ex_count == heur_count

    def test_exhaustive_self_consistency(self, system_2d_simple):
        """All FPs found by exhaustive search pass self-consistency."""
        from dynamic.analysis.exhaustive_search import main_exhaustive
        from dynamic.analysis.scyfi_helpers import latent_step

        A, W, h = system_2d_simple
        result = main_exhaustive(A, W, h, 1)
        for traj in result[0][0]:
            z_star = traj[0]
            z_next = latent_step(z_star, A, W, h)
            assert torch.norm(z_next - z_star).item() < 1e-6

    def test_exhaustive_eigvals_returned(self, system_2d_simple):
        """Eigenvalues are returned for each cycle."""
        from dynamic.analysis.exhaustive_search import main_exhaustive

        A, W, h = system_2d_simple
        result = main_exhaustive(A, W, h, 2)
        for order_idx in range(2):
            assert len(result[1][order_idx]) == len(result[0][order_idx])
