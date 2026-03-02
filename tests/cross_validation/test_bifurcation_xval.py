"""Cross-validation: bifurcation detection comparisons.

Tests compare_stability and distance metrics between Julia and Python.
These functions are deterministic, so results should match exactly.
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
# compare_stability
# ======================================================================
class TestCompareStabilityXval:
    """Stability comparison must match exactly."""

    def test_stable_match(self, jw):
        """Both agree on matching stability."""
        from dynamic.analysis.bifurcation import compare_stability

        eigvals = np.array([0.5, 0.3])
        neighbours = [
            np.array([0.4, 0.6]),
            np.array([1.5, 0.3]),
        ]
        py_result = compare_stability(eigvals, neighbours)

        # Julia side
        jl_result_raw = jw.jl.seval(
            """(ev, n1, n2) -> begin
                SCYFI.compare_stability(ev, [n1, n2])
            end"""
        )(eigvals, neighbours[0], neighbours[1])

        if py_result is None:
            assert jl_result_raw is None or jw.jl.seval("isnothing")(jl_result_raw)
        else:
            jl_result = [int(x) for x in jl_result_raw]
            # Julia is 1-indexed, convert to 0-indexed
            jl_result_0 = [x - 1 for x in jl_result]
            assert sorted(py_result) == sorted(jl_result_0)

    def test_no_match(self, jw):
        """Both agree on no match → None/nothing."""
        from dynamic.analysis.bifurcation import compare_stability

        eigvals = np.array([0.5, 0.3])
        neighbours = [np.array([1.5, 1.3])]
        py_result = compare_stability(eigvals, neighbours)

        jl_result_raw = jw.jl.seval(
            """(ev, n1) -> begin
                SCYFI.compare_stability(ev, [n1])
            end"""
        )(eigvals, neighbours[0])

        assert py_result is None
        assert jl_result_raw is None or jw.jl.seval("isnothing")(jl_result_raw)


# ======================================================================
# Distance metrics
# ======================================================================
class TestDistanceMetricsXval:
    """State-space and eigenvalue distances must match."""

    def test_state_space_distance(self, jw):
        """Minimal state-space distances match."""
        from dynamic.analysis.bifurcation import (
            get_minimal_state_space_distances,
        )

        cycle = [torch.tensor([1.0, 2.0], dtype=torch.float64)]
        neighbours = [
            [torch.tensor([1.1, 2.1], dtype=torch.float64)],
            [torch.tensor([5.0, 5.0], dtype=torch.float64)],
        ]
        py_dists = get_minimal_state_space_distances(cycle, neighbours)

        # Call Julia
        jl_dists = jw.jl.seval(
            """(c, n1, n2) -> begin
                SCYFI.get_minimal_state_space_distances(c, [n1, n2])
            end"""
        )(
            [cycle[0].numpy()],
            [neighbours[0][0].numpy()],
            [neighbours[1][0].numpy()],
        )
        jl_dists_py = [float(jl_dists[i]) for i in range(len(jl_dists))]

        np.testing.assert_allclose(py_dists, jl_dists_py, atol=1e-10)

    def test_eigenvalue_distance(self, jw):
        """Minimal eigenvalue distances match."""
        from dynamic.analysis.bifurcation import (
            get_minimal_eigenvalue_distances,
        )

        eigvals = np.array([0.5, -0.3])
        neighbours = [np.array([0.6, -0.2]), np.array([2.0, 3.0])]
        py_dists = get_minimal_eigenvalue_distances(eigvals, neighbours)

        jl_dists = jw.jl.seval(
            """(ev, n1, n2) -> begin
                SCYFI.get_minimal_eigenvalue_distances(ev, [n1, n2])
            end"""
        )(eigvals, neighbours[0], neighbours[1])
        jl_dists_py = [float(jl_dists[i]) for i in range(len(jl_dists))]

        np.testing.assert_allclose(py_dists, jl_dists_py, atol=1e-10)
