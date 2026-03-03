"""Analysis tools: fixed points, manifolds, quality metrics."""

from dynamic.analysis.backtracking import (
    backtrack_trajectory,
    backward_step,
    try_bitflips,
    verify_forward,
)
from dynamic.analysis.bifurcation import (
    compare_stability,
    create_grid_data,
    find_bifurcations_parameter_grid,
    find_bifurcations_trajectory,
    get_combined_state_space_eigenvalue_distance,
    get_minimal_eigenvalue_distances,
    get_minimal_state_space_distances,
)
from dynamic.analysis.fallback import fallback_manifold_detection
from dynamic.analysis.homoclinic import find_homoclinic_intersections
from dynamic.analysis.lyapunov import compute_lyapunov_exponents
from dynamic.analysis.manifolds import (
    ManifoldSegment,
    construct_manifold,
    sample_on_manifold,
)
from dynamic.analysis.quality import delta_sigma, delta_sigma_statistic
from dynamic.analysis.scyfi import (
    Cycle,
    FixedPoint,
    find_cycles,
    find_cycles_sh,
    find_fixed_points,
)
from dynamic.analysis.subregions import classify_point

__all__ = [
    "Cycle",
    "FixedPoint",
    "ManifoldSegment",
    "backtrack_trajectory",
    "backward_step",
    "classify_point",
    "compare_stability",
    "compute_lyapunov_exponents",
    "construct_manifold",
    "create_grid_data",
    "delta_sigma",
    "delta_sigma_statistic",
    "fallback_manifold_detection",
    "find_bifurcations_parameter_grid",
    "find_bifurcations_trajectory",
    "find_cycles",
    "find_cycles_sh",
    "find_fixed_points",
    "find_homoclinic_intersections",
    "get_combined_state_space_eigenvalue_distance",
    "get_minimal_eigenvalue_distances",
    "get_minimal_state_space_distances",
    "sample_on_manifold",
    "try_bitflips",
    "verify_forward",
]
