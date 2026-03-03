"""Benchmark all SCYFI variants: timing + correctness.

For each (dim, order) configuration:
  1. Run each variant and record wall-clock time
  2. Count cycles found per variant
  3. Validate all found cycles via forward simulation
  4. Cross-check: compare cycle sets against exhaustive ground truth (where feasible)
  5. Skip variants that exceed 3s timeout
"""
# ruff: noqa: E501

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynamic.analysis.scyfi_exhaustive import find_cycles_exhaustive  # noqa: E402
from dynamic.analysis.scyfi_fused import find_cycles_fused  # noqa: E402
from dynamic.analysis.scyfi_hybrid import find_cycles_hybrid  # noqa: E402
from dynamic.analysis.scyfi_newton import find_cycles_newton  # noqa: E402
from dynamic.analysis.scyfi_power import find_cycles_power  # noqa: E402
from dynamic.analysis.scyfi_schur import find_cycles_schur  # noqa: E402
from dynamic.analysis.scyfi_vectorised import find_cycles_vectorised  # noqa: E402
from dynamic.analysis.scyfi_woodbury import find_cycles_woodbury  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TIMEOUT = 3.0  # seconds


def _cycle_key(z: torch.Tensor, scale: int = 1000) -> tuple[int, ...]:
    """Quantised key for a single point."""
    return tuple(torch.round(z * scale).long().tolist())


def _cycle_set(
    cycles_per_order: list[list[list[torch.Tensor]]],
) -> set[tuple[int, ...]]:
    """Build a set of quantised cycle-point keys from all orders."""
    keys = set()
    for order_cycles in cycles_per_order:
        for traj in order_cycles:
            for pt in traj:
                keys.add(_cycle_key(pt))
    return keys


def _count_cycles(cycles_per_order: list[list[list[torch.Tensor]]]) -> list[int]:
    """Count cycles found per order."""
    return [len(c) for c in cycles_per_order]


def _validate_cycles(
    A: torch.Tensor,
    W: torch.Tensor,
    h: torch.Tensor,
    cycles_per_order: list[list[list[torch.Tensor]]],
    atol: float = 1e-5,
) -> tuple[int, int]:
    """Validate all cycles via forward simulation. Returns (n_valid, n_total)."""
    n_valid = 0
    n_total = 0
    for order_idx, order_cycles in enumerate(cycles_per_order):
        order = order_idx + 1
        for traj in order_cycles:
            n_total += 1
            ok = True
            for j in range(order):
                z = traj[j]
                z_next = A @ z + W @ torch.clamp(z, min=0.0) + h
                expected = traj[(j + 1) % order]
                if not torch.allclose(z_next, expected, atol=atol):
                    ok = False
                    break
            if ok:
                n_valid += 1
    return n_valid, n_total


def _overlap(found: set, ground_truth: set) -> tuple[int, int, int]:
    """Returns (overlap, found_only, gt_only)."""
    return (
        len(found & ground_truth),
        len(found - ground_truth),
        len(ground_truth - found),
    )


def bench_and_validate(fn, A, W, h, max_order, reps=5, **kw):
    """Run a variant, measure time, validate results."""
    times = []
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(A, W, h, max_order, **kw)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
        if elapsed > TIMEOUT:
            return float("inf"), result
    return float(np.median(times)), result


# ---------------------------------------------------------------------------
# Julia setup
# ---------------------------------------------------------------------------
def setup_julia():
    """Try to set up Julia. Returns bench function or None."""
    try:
        juliaup_bin = os.path.expanduser("~/.juliaup/bin")
        os.environ["PATH"] = juliaup_bin + ":" + os.environ.get("PATH", "")
        os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
        from juliacall import Main as jl

        xval_dir = os.path.join(
            os.path.dirname(__file__), "..", "tests", "cross_validation"
        )
        scyfi_src = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "reference", "SCYFI", "src")
        )
        bifur = os.path.abspath(
            os.path.join(xval_dir, "helpers_bifurcation_stripped.jl")
        )
        boot = os.path.abspath(os.path.join(xval_dir, "scyfi_bootstrap.jl"))
        jl.seval(
            'import Pkg; try; using Distributions; catch; Pkg.add("Distributions"); using Distributions; end'
        )
        jl.seval(
            f"module SCYFI; using LinearAlgebra; using Random; using Distributions; "
            f'include("{scyfi_src}/utilities/helpers.jl"); '
            f'include("{scyfi_src}/utilities/exhaustive_search.jl"); '
            f'include("{bifur}"); include("{boot}"); end'
        )

        def julia_bench(A_np, W_np, h_np, mo, outer=10, inner=20):
            return float(
                jl.seval(
                    "(A,W,h,mo,o,i)->begin; A_jl=Matrix{Float64}(A); W_jl=Matrix{Float64}(W); "
                    "h_jl=Vector{Float64}(h); "
                    "SCYFI.find_cycles(A_jl,W_jl,h_jl,mo,outer_loop_iterations=o,inner_loop_iterations=i); "
                    "median([(@elapsed SCYFI.find_cycles(A_jl,W_jl,h_jl,mo,"
                    "outer_loop_iterations=o,inner_loop_iterations=i)) for _ in 1:5])*1000; end"
                )(A_np, W_np, h_np, mo, outer, inner)
            )

        return julia_bench
    except Exception as e:
        print(f"Julia not available: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    julia_bench = setup_julia()

    # Warmup
    A_w = torch.diag(torch.randn(2, dtype=torch.float64))
    W_w = torch.randn(2, 2, dtype=torch.float64)
    h_w = torch.randn(2, dtype=torch.float64)
    find_cycles_vectorised(
        A_w,
        W_w,
        h_w,
        1,
        outer_loop_iterations=1,
        inner_loop_iterations=2,
        batch_size=4,
        compiled=True,
    )

    dims = [2, 5, 10, 20, 50]
    orders = [1, 4, 8, 15]

    exceeded: dict[tuple[str, int], bool] = {}

    variants = [
        (
            "Fused",
            find_cycles_fused,
            {"outer_loop_iterations": 10, "inner_loop_iterations": 20},
        ),
        (
            "VecC",
            find_cycles_vectorised,
            {
                "outer_loop_iterations": 2,
                "inner_loop_iterations": 3,
                "batch_size": 64,
                "compiled": True,
            },
        ),
        ("Exhst", find_cycles_exhaustive, {"max_systems": 100_000}),
        ("Power", find_cycles_power, {}),
        ("Newton", find_cycles_newton, {}),
        ("Hybrid", find_cycles_hybrid, {}),
        ("Wdbury", find_cycles_woodbury, {}),
        (
            "Schur",
            find_cycles_schur,
            {"B": 64, "outer_loop_iterations": 2, "inner_loop_iterations": 3},
        ),
    ]

    # Header
    header_names = [v[0] for v in variants]
    if julia_bench:
        header_names.append("Julia")
    col_w = 16
    print()
    print(f"{'Dim':>4} {'Ord':>4}  ", end="")
    for n in header_names:
        print(f"{n:>{col_w}}", end="")
    print()
    print("=" * (10 + col_w * len(header_names)))

    for dim in dims:
        for order in orders:
            torch.manual_seed(42)
            A = torch.diag(torch.randn(dim, dtype=torch.float64))
            W = torch.randn(dim, dim, dtype=torch.float64) * 0.3
            W.fill_diagonal_(0.0)
            h = torch.randn(dim, dtype=torch.float64)

            # Ground truth (exhaustive where feasible)
            n_sys = (1 << dim) ** order
            gt_set = None
            gt_counts = None
            if n_sys <= 100_000:
                _, gt_result = bench_and_validate(
                    find_cycles_exhaustive, A, W, h, order, reps=1, max_systems=100_000
                )
                if gt_result is not None:
                    gt_set = _cycle_set(gt_result[0])

            cells = []
            for name, fn, kw in variants:
                key = (name, dim)

                # Woodbury only k=1
                if name == "Wdbury" and order > 1:
                    cells.append("—")
                    continue

                # Skip if exceeded
                if key in exceeded:
                    cells.append("SKIP")
                    continue

                # Exhaustive feasibility
                if name == "Exhst" and n_sys > 100_000:
                    cells.append("—")
                    continue

                torch.manual_seed(42)
                t_ms, result = bench_and_validate(fn, A, W, h, order, **kw)

                if t_ms == float("inf"):
                    exceeded[key] = True
                    cells.append("SKIP")
                    continue
                if t_ms > TIMEOUT * 1000:
                    exceeded[key] = True

                cycles, eigvals = result
                counts = _count_cycles(cycles)
                n_valid, n_total = _validate_cycles(A, W, h, cycles)

                # Build cell: "12.3ms 3c ✓"
                if t_ms >= 1000:
                    time_str = f"{t_ms / 1000:.1f}s"
                else:
                    time_str = f"{t_ms:.1f}ms"

                total_c = sum(counts)
                valid_str = "✓" if n_valid == n_total else f"✗{n_total - n_valid}"

                # Diff against ground truth
                diff_str = ""
                if gt_set is not None and name != "Exhst":
                    found_set = _cycle_set(cycles)
                    overlap, extra, missing = _overlap(found_set, gt_set)
                    if missing == 0 and extra == 0:
                        diff_str = " =="
                    elif missing == 0:
                        diff_str = f" +{extra}"
                    elif extra == 0:
                        diff_str = f" -{missing}"
                    else:
                        diff_str = f" +{extra}-{missing}"

                cell = f"{time_str} {total_c}c {valid_str}{diff_str}"
                cells.append(cell)

            # Julia
            if julia_bench:
                jl_t = julia_bench(A.numpy(), W.numpy(), h.numpy(), order, 10, 20)
                if jl_t >= 1000:
                    cells.append(f"{jl_t / 1000:.1f}s")
                else:
                    cells.append(f"{jl_t:.1f}ms")

            # Print row
            print(f"{dim:4d} {order:4d}  ", end="")
            for cell in cells:
                print(f"{cell:>{col_w}}", end="")
            print()

        print("-" * (10 + col_w * len(header_names)))

    print()
    print("Legend: <time> <cycles_found>c <valid> [<diff vs exhaustive>]")
    print("  ✓ = all cycles pass forward-simulation check")
    print("  == = exact match with exhaustive ground truth")
    print("  +N = N extra cycle points found (not in exhaustive)")
    print("  -N = N cycle points missed (in exhaustive but not found)")
    print("  SKIP = exceeded 3s timeout, — = infeasible/not applicable")


if __name__ == "__main__":
    main()
