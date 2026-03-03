"""Performance benchmarks for SCYFI functions.

Compares Python/PyTorch vs Julia runtime across varying dimensions and
cycle orders. Uses Julia-native timing (@elapsed) for helpers to avoid
FFI overhead distortion.

Run with:
    python benchmarks/bench_scyfi.py           # Python only
    python benchmarks/bench_scyfi.py --julia   # Python + Julia
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Add project to path
# ---------------------------------------------------------------------------
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src")
)


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    """Single benchmark result."""

    name: str
    dim: int
    order: int
    py_time_ms: float
    jl_time_ms: float | None = None

    @property
    def speedup(self) -> str:
        if self.jl_time_ms is None or self.jl_time_ms == 0:
            return "—"
        ratio = self.py_time_ms / self.jl_time_ms
        if ratio >= 1:
            return f"Py {ratio:.1f}× slower"
        return f"Py {1/ratio:.1f}× faster"


def _timeit(fn, warmup: int = 2, repeat: int = 5) -> float:
    """Time a callable, return median time in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(np.median(times))


def _print_table(results: list[BenchResult], title: str) -> None:
    """Print results as a formatted table."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    has_julia = any(r.jl_time_ms is not None for r in results)

    if has_julia:
        header = (
            f"{'Function':<35} {'Dim':>4} {'Ord':>4} "
            f"{'Python ms':>10} {'Julia ms':>10} {'Comparison':>20}"
        )
        print(header)
        print("-" * 90)
        for r in results:
            jl = f"{r.jl_time_ms:10.3f}" if r.jl_time_ms is not None else "       —"
            spd = r.speedup if r.jl_time_ms is not None else "—"
            print(
                f"{r.name:<35} {r.dim:4d} {r.order:4d} "
                f"{r.py_time_ms:10.3f} {jl} {spd:>20}"
            )
    else:
        header = (
            f"{'Function':<35} {'Dim':>4} {'Ord':>4} {'Python ms':>10}"
        )
        print(header)
        print("-" * 60)
        for r in results:
            print(
                f"{r.name:<35} {r.dim:4d} {r.order:4d} "
                f"{r.py_time_ms:10.3f}"
            )
    print()


# ---------------------------------------------------------------------------
# Parameter generators
# ---------------------------------------------------------------------------
def _make_plrnn_params(dim: int, dtype=torch.float64):
    """Generate random PLRNN params (A, W, h)."""
    A = torch.diag(torch.randn(dim, dtype=dtype))
    W = torch.randn(dim, dim, dtype=dtype) * 0.3
    W.fill_diagonal_(0.0)
    h = torch.randn(dim, dtype=dtype)
    return A, W, h


def _make_shplrnn_params(dim: int, hidden_dim: int, dtype=torch.float64):
    """Generate random shPLRNN params."""
    A = torch.randn(dim, dtype=dtype)
    W1 = torch.randn(dim, hidden_dim, dtype=dtype) * 0.3
    W2 = torch.randn(hidden_dim, dim, dtype=dtype) * 0.3
    h1 = torch.randn(dim, dtype=dtype)
    h2 = torch.randn(hidden_dim, dtype=dtype)
    return A, W1, W2, h1, h2


def _make_d_list(dim: int, order: int, dtype=torch.float64):
    """Generate random D-list."""
    from dynamic.analysis.scyfi_helpers import construct_relu_matrix_list

    return construct_relu_matrix_list(dim, order, dtype=dtype)


# ---------------------------------------------------------------------------
# Julia-native timing helpers
# ---------------------------------------------------------------------------
def _julia_timed_helpers(jl, dim, order, A_np, W_np, h_np, D_np):
    """Run helper benchmarks entirely inside Julia using @elapsed.

    This avoids FFI overhead and gives true Julia computation time.
    Returns dict of function_name -> time_ms.
    """
    result = jl.seval("""
        (A, W, h, D, dim, order) -> begin
            A_jl = Matrix{Float64}(A)
            W_jl = Matrix{Float64}(W)
            h_jl = Vector{Float64}(h)
            D_jl = Array{Float64,3}(D)

            times = Dict{String, Float64}()

            # construct_relu_matrix
            q = Int128(2^dim - 1)
            # Warmup
            for _ in 1:3; SCYFI.construct_relu_matrix(q, dim); end
            t = median([(@elapsed SCYFI.construct_relu_matrix(q, dim))
                        for _ in 1:20])
            times["construct_relu_matrix"] = t * 1000

            # get_factor_in_front_of_z
            for _ in 1:3; SCYFI.get_factor_in_front_of_z(A_jl, W_jl, D_jl, order); end
            t = median([(@elapsed SCYFI.get_factor_in_front_of_z(A_jl, W_jl, D_jl, order))
                        for _ in 1:20])
            times["get_factor_in_front_of_z"] = t * 1000

            # get_factor_in_front_of_h (uses D_list[:,:,2:end])
            D_sliced = order > 1 ? D_jl[:,:,2:end] : D_jl[:,:,1:0]
            for _ in 1:3; SCYFI.get_factor_in_front_of_h(A_jl, W_jl, D_sliced, order); end
            t = median([(@elapsed SCYFI.get_factor_in_front_of_h(A_jl, W_jl, D_sliced, order))
                        for _ in 1:20])
            times["get_factor_in_front_of_h"] = t * 1000

            # get_cycle_point_candidate
            for _ in 1:3; SCYFI.get_cycle_point_candidate(A_jl, W_jl, D_jl, h_jl, order); end
            t = median([(@elapsed SCYFI.get_cycle_point_candidate(A_jl, W_jl, D_jl, h_jl, order))
                        for _ in 1:20])
            times["get_cycle_point_candidate"] = t * 1000

            # get_eigvals
            for _ in 1:3; SCYFI.get_eigvals(A_jl, W_jl, D_jl, order); end
            t = median([(@elapsed SCYFI.get_eigvals(A_jl, W_jl, D_jl, order))
                        for _ in 1:20])
            times["get_eigvals"] = t * 1000

            # latent_step
            z = randn(dim)
            for _ in 1:3; SCYFI.latent_step(z, A_jl, W_jl, h_jl); end
            t = median([(@elapsed SCYFI.latent_step(z, A_jl, W_jl, h_jl))
                        for _ in 1:20])
            times["latent_step"] = t * 1000

            # get_latent_time_series (100 steps)
            for _ in 1:3; SCYFI.get_latent_time_series(100, A_jl, W_jl, h_jl, dim, z_0=z); end
            t = median([(@elapsed SCYFI.get_latent_time_series(100, A_jl, W_jl, h_jl, dim, z_0=z))
                        for _ in 1:20])
            times["get_latent_time_series(100)"] = t * 1000

            return times
        end
    """)(A_np, W_np, h_np, D_np, dim, order)

    return {str(k): float(v) for k, v in result.items()}


def _julia_timed_find_cycles(jl, A_np, W_np, h_np, max_order, outer, inner):
    """Time find_cycles inside Julia using @elapsed."""
    t_ms = jl.seval("""
        (A, W, h, max_order, outer, inner) -> begin
            A_jl = Matrix{Float64}(A)
            W_jl = Matrix{Float64}(W)
            h_jl = Vector{Float64}(h)

            # Warmup
            SCYFI.find_cycles(A_jl, W_jl, h_jl, max_order,
                outer_loop_iterations=outer, inner_loop_iterations=inner)

            t = median([
                (@elapsed SCYFI.find_cycles(A_jl, W_jl, h_jl, max_order,
                    outer_loop_iterations=outer, inner_loop_iterations=inner))
                for _ in 1:3
            ])
            return t * 1000
        end
    """)(A_np, W_np, h_np, max_order, outer, inner)
    return float(t_ms)


def _julia_timed_find_cycles_sh(
    jl, A_np, W1_np, W2_np, h1_np, h2_np, max_order, outer, inner,
):
    """Time find_cycles for shPLRNN inside Julia using @elapsed."""
    t_ms = jl.seval("""
        (A, W1, W2, h1, h2, max_order, outer, inner) -> begin
            A_jl = Vector{Float64}(A)
            W1_jl = Matrix{Float64}(W1)
            W2_jl = Matrix{Float64}(W2)
            h1_jl = Vector{Float64}(h1)
            h2_jl = Vector{Float64}(h2)

            # Warmup
            SCYFI.find_cycles(A_jl, W1_jl, W2_jl, h1_jl, h2_jl, max_order,
                outer_loop_iterations=outer, inner_loop_iterations=inner)

            t = median([
                (@elapsed SCYFI.find_cycles(A_jl, W1_jl, W2_jl, h1_jl, h2_jl, max_order,
                    outer_loop_iterations=outer, inner_loop_iterations=inner))
                for _ in 1:3
            ])
            return t * 1000
        end
    """)(A_np, W1_np, W2_np, h1_np, h2_np, max_order, outer, inner)
    return float(t_ms)


# ---------------------------------------------------------------------------
# Helper benchmarks
# ---------------------------------------------------------------------------
def bench_helpers(
    dims: list[int], orders: list[int], jl=None,
) -> list[BenchResult]:
    """Benchmark core helper functions."""
    from dynamic.analysis.scyfi_helpers import (
        construct_relu_matrix,
        get_cycle_point_candidate,
        get_eigvals,
        get_factor_in_front_of_h,
        get_factor_in_front_of_z,
        get_latent_time_series,
        latent_step,
    )

    results = []

    for dim in dims:
        A, W, h = _make_plrnn_params(dim)

        for order in orders:
            D_list = _make_d_list(dim, order)

            # Julia native timing (all at once, no FFI overhead)
            jl_times = None
            if jl:
                D_np = D_list.numpy().transpose(1, 2, 0).copy()
                jl_times = _julia_timed_helpers(
                    jl, dim, order,
                    A.numpy(), W.numpy(), h.numpy(), D_np,
                )

            # construct_relu_matrix
            q = 2**dim - 1
            py_t = _timeit(
                lambda: construct_relu_matrix(q, dim, dtype=A.dtype)
            )
            jl_t = jl_times["construct_relu_matrix"] if jl_times else None
            results.append(
                BenchResult("construct_relu_matrix", dim, order, py_t, jl_t)
            )

            # get_factor_in_front_of_z
            py_t = _timeit(
                lambda: get_factor_in_front_of_z(A, W, D_list, order)
            )
            jl_t = jl_times["get_factor_in_front_of_z"] if jl_times else None
            results.append(
                BenchResult("get_factor_in_front_of_z", dim, order, py_t, jl_t)
            )

            # get_factor_in_front_of_h
            py_t = _timeit(
                lambda: get_factor_in_front_of_h(A, W, D_list, order)
            )
            jl_t = jl_times["get_factor_in_front_of_h"] if jl_times else None
            results.append(
                BenchResult("get_factor_in_front_of_h", dim, order, py_t, jl_t)
            )

            # get_cycle_point_candidate
            py_t = _timeit(
                lambda: get_cycle_point_candidate(A, W, D_list, h, order)
            )
            jl_t = jl_times["get_cycle_point_candidate"] if jl_times else None
            results.append(
                BenchResult("get_cycle_point_candidate", dim, order, py_t, jl_t)
            )

            # get_eigvals
            py_t = _timeit(
                lambda: get_eigvals(A, W, D_list, order)
            )
            jl_t = jl_times["get_eigvals"] if jl_times else None
            results.append(
                BenchResult("get_eigvals", dim, order, py_t, jl_t)
            )

        # latent_step + time_series (once per dim, order=1)
        z = torch.randn(dim, dtype=A.dtype)
        py_t = _timeit(lambda: latent_step(z, A, W, h))
        jl_t = None
        if jl:
            D_o1 = _make_d_list(dim, 1)
            D_np = D_o1.numpy().transpose(1, 2, 0).copy()
            jl_times_1 = _julia_timed_helpers(
                jl, dim, 1, A.numpy(), W.numpy(), h.numpy(), D_np,
            )
            jl_t = jl_times_1["latent_step"]
        results.append(BenchResult("latent_step", dim, 1, py_t, jl_t))

        py_t = _timeit(
            lambda: get_latent_time_series(100, A, W, h, dim, z_0=z)
        )
        jl_t = jl_times_1["get_latent_time_series(100)"] if jl else None
        results.append(
            BenchResult("get_latent_time_series(100)", dim, 1, py_t, jl_t)
        )

    return results


# ---------------------------------------------------------------------------
# Algorithm-level benchmarks
# ---------------------------------------------------------------------------
def bench_algorithm(
    dims: list[int], orders: list[int], jl=None,
) -> list[BenchResult]:
    """Benchmark top-level find_cycles."""
    from dynamic.analysis.scyfi import find_cycles, find_cycles_sh

    results = []
    outer, inner = 10, 20

    for dim in dims:
        A, W, h = _make_plrnn_params(dim)

        for max_order in orders:
            py_t = _timeit(
                lambda mo=max_order: find_cycles(
                    A, W, h, mo,
                    outer_loop_iterations=outer,
                    inner_loop_iterations=inner,
                ),
                warmup=1, repeat=3,
            )
            jl_t = None
            if jl:
                jl_t = _julia_timed_find_cycles(
                    jl, A.numpy(), W.numpy(), h.numpy(),
                    max_order, outer, inner,
                )
            results.append(
                BenchResult("find_cycles (PLRNN)", dim, max_order, py_t, jl_t)
            )

    # shPLRNN — smaller dims only (pool can be very large)
    sh_dims = [d for d in dims if d <= 5]
    for dim in sh_dims:
        hidden_dim = 3 * dim
        A_sh, W1, W2, h1, h2 = _make_shplrnn_params(dim, hidden_dim)

        for max_order in [o for o in orders if o <= 4]:
            py_t = _timeit(
                lambda mo=max_order: find_cycles_sh(
                    A_sh, W1, W2, h1, h2, mo,
                    outer_loop_iterations=outer,
                    inner_loop_iterations=inner,
                ),
                warmup=1, repeat=3,
            )
            jl_t = None
            if jl:
                jl_t = _julia_timed_find_cycles_sh(
                    jl,
                    A_sh.numpy(), W1.numpy(), W2.numpy(),
                    h1.numpy(), h2.numpy(),
                    max_order, outer, inner,
                )
            results.append(
                BenchResult(
                    f"find_cycles (shPLRNN h={hidden_dim})",
                    dim, max_order, py_t, jl_t,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Optimised algorithm benchmarks
# ---------------------------------------------------------------------------
def bench_algorithm_fast(
    dims: list[int], orders: list[int],
) -> list[BenchResult]:
    """Benchmark optimised find_cycles variants vs reference."""
    from dynamic.analysis.scyfi import find_cycles, find_cycles_sh
    from dynamic.analysis.scyfi_fast import (
        find_cycles_fast,
        find_cycles_sh_fast,
    )

    results = []
    outer, inner = 10, 20

    for dim in dims:
        A, W, h = _make_plrnn_params(dim)

        for max_order in orders:
            ref_t = _timeit(
                lambda mo=max_order: find_cycles(
                    A, W, h, mo,
                    outer_loop_iterations=outer,
                    inner_loop_iterations=inner,
                ),
                warmup=1, repeat=3,
            )
            fast_t = _timeit(
                lambda mo=max_order: find_cycles_fast(
                    A, W, h, mo,
                    outer_loop_iterations=outer,
                    inner_loop_iterations=inner,
                ),
                warmup=1, repeat=3,
            )
            batch_t = _timeit(
                lambda mo=max_order: find_cycles_fast(
                    A, W, h, mo,
                    outer_loop_iterations=outer,
                    inner_loop_iterations=inner,
                    batched=True, batch_size=32,
                ),
                warmup=1, repeat=3,
            )
            results.append(BenchResult("reference", dim, max_order, ref_t))
            results.append(BenchResult("fast (sequential)", dim, max_order, fast_t))
            results.append(BenchResult("fast (batched B=32)", dim, max_order, batch_t))

    sh_dims = [d for d in dims if d <= 5]
    for dim in sh_dims:
        hidden_dim = 3 * dim
        A_sh, W1, W2, h1, h2 = _make_shplrnn_params(dim, hidden_dim)
        for max_order in [o for o in orders if o <= 4]:
            ref_t = _timeit(
                lambda mo=max_order: find_cycles_sh(
                    A_sh, W1, W2, h1, h2, mo,
                    outer_loop_iterations=outer,
                    inner_loop_iterations=inner,
                ),
                warmup=1, repeat=3,
            )
            fast_t = _timeit(
                lambda mo=max_order: find_cycles_sh_fast(
                    A_sh, W1, W2, h1, h2, mo,
                    outer_loop_iterations=outer,
                    inner_loop_iterations=inner,
                ),
                warmup=1, repeat=3,
            )
            results.append(
                BenchResult(f"shPLRNN ref (h={hidden_dim})", dim, max_order, ref_t)
            )
            results.append(
                BenchResult(f"shPLRNN fast (h={hidden_dim})", dim, max_order, fast_t)
            )

    return results


# ---------------------------------------------------------------------------
# Bifurcation benchmarks
# ---------------------------------------------------------------------------
def bench_bifurcation(dims: list[int]) -> list[BenchResult]:
    """Benchmark bifurcation detection functions."""
    from dynamic.analysis.bifurcation import (
        compare_stability,
        get_combined_state_space_eigenvalue_distance,
        get_minimal_eigenvalue_distances,
        get_minimal_state_space_distances,
    )

    results = []

    for dim in dims:
        cycle = [torch.randn(dim, dtype=torch.float64)]
        n_neighbours = 50
        neighbours = [
            [torch.randn(dim, dtype=torch.float64)]
            for _ in range(n_neighbours)
        ]
        eigvals = np.random.randn(dim)
        eigvals_nbrs = [np.random.randn(dim) for _ in range(n_neighbours)]

        py_t = _timeit(lambda: compare_stability(eigvals, eigvals_nbrs))
        results.append(
            BenchResult("compare_stability", dim, n_neighbours, py_t)
        )

        py_t = _timeit(
            lambda: get_minimal_state_space_distances(cycle, neighbours)
        )
        results.append(
            BenchResult("min_state_space_dist", dim, n_neighbours, py_t)
        )

        py_t = _timeit(
            lambda: get_minimal_eigenvalue_distances(eigvals, eigvals_nbrs)
        )
        results.append(
            BenchResult("min_eigenvalue_dist", dim, n_neighbours, py_t)
        )

        py_t = _timeit(
            lambda: get_combined_state_space_eigenvalue_distance(
                cycle, neighbours, eigvals, eigvals_nbrs
            )
        )
        results.append(
            BenchResult("combined_ss_eig_dist", dim, n_neighbours, py_t)
        )

    return results


# ---------------------------------------------------------------------------
# Julia setup
# ---------------------------------------------------------------------------
def _setup_julia():
    """Load Julia SCYFI module, return jl handle."""
    juliaup_bin = os.path.expanduser("~/.juliaup/bin")
    if juliaup_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = juliaup_bin + ":" + os.environ.get("PATH", "")

    from juliacall import Main as jl

    xval_dir = os.path.join(
        os.path.dirname(__file__), "..", "tests", "cross_validation"
    )
    scyfi_src = os.path.abspath(
        os.path.join(xval_dir, "..", "..", "reference", "SCYFI", "src")
    )
    bifur_stripped = os.path.abspath(
        os.path.join(xval_dir, "helpers_bifurcation_stripped.jl")
    )
    bootstrap = os.path.abspath(
        os.path.join(xval_dir, "scyfi_bootstrap.jl")
    )

    jl.seval("""
        import Pkg
        try; using Distributions; catch; Pkg.add("Distributions"); using Distributions; end
    """)

    jl.seval(f"""
        module SCYFI
            using LinearAlgebra
            using Random
            using Distributions
            include("{scyfi_src}/utilities/helpers.jl")
            include("{scyfi_src}/utilities/exhaustive_search.jl")
            include("{bifur_stripped}")
            include("{bootstrap}")
        end
    """)
    jl.seval("using LinearAlgebra, Statistics")
    return jl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SCYFI performance benchmarks"
    )
    parser.add_argument(
        "--julia", action="store_true",
        help="Include Julia benchmarks (requires juliacall)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode — fewer dims/orders"
    )
    args = parser.parse_args()

    if args.quick:
        dims = [2, 5, 10]
        orders = [1, 4]
    else:
        dims = [2, 5, 10, 20, 50]
        orders = [1, 4, 8, 16]

    jl = None
    if args.julia:
        print("Loading Julia SCYFI module...")
        jl = _setup_julia()
        print("Julia ready.\n")

    # 1. Helper benchmarks
    print("Running helper benchmarks...")
    helper_results = bench_helpers(dims, orders, jl=jl)
    _print_table(helper_results, "Helper Functions")

    # 2. Algorithm benchmarks (reference + Julia)
    algo_dims = [d for d in dims if d <= 20]
    algo_orders = [o for o in orders if o <= 8]
    print("Running algorithm benchmarks (reference)...")
    algo_results = bench_algorithm(algo_dims, algo_orders, jl=jl)
    _print_table(algo_results, "Algorithm — find_cycles (reference)")

    # 3. Optimised algorithm benchmarks
    print("Running algorithm benchmarks (optimised)...")
    fast_results = bench_algorithm_fast(algo_dims, algo_orders)
    _print_table(fast_results, "Algorithm — find_cycles (ref vs fast)")

    # 4. Bifurcation benchmarks
    print("Running bifurcation benchmarks...")
    bifur_results = bench_bifurcation(dims)
    _print_table(bifur_results, "Bifurcation Detection (50 neighbours)")

    # Summary
    all_py = helper_results + algo_results + bifur_results
    total_py = sum(r.py_time_ms for r in all_py)
    print(f"Total Python reference time: {total_py:.1f} ms")
    if jl:
        jl_items = [r for r in all_py if r.jl_time_ms is not None]
        if jl_items:
            total_jl = sum(r.jl_time_ms for r in jl_items)
            total_py_matched = sum(r.py_time_ms for r in jl_items)
            print(f"Total Julia time (matched):  {total_jl:.1f} ms")
            print(
                f"Overall ratio: Python ref is "
                f"{total_py_matched/total_jl:.1f}× "
                f"{'slower' if total_py_matched > total_jl else 'faster'}"
            )


if __name__ == "__main__":
    main()

