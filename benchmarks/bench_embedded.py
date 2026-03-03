"""Benchmark all SCYFI variants on embedded datasets with GT validation.

Outputs proper markdown tables with separate columns per metric.
Includes: Original, VecC, Newton, Hybrid, Schur, Julia.
"""
# ruff: noqa: E501

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.datasets import build_all_datasets  # noqa: E402
from dynamic.analysis.scyfi import find_cycles as find_cycles_original  # noqa: E402
from dynamic.analysis.scyfi_hybrid import find_cycles_hybrid  # noqa: E402
from dynamic.analysis.scyfi_newton import find_cycles_newton  # noqa: E402
from dynamic.analysis.scyfi_schur import find_cycles_schur  # noqa: E402
from dynamic.analysis.scyfi_vectorised import find_cycles_vectorised  # noqa: E402

TIMEOUT = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cycle_key(z: torch.Tensor, scale: int = 1000) -> tuple[int, ...]:
    return tuple(torch.round(z * scale).long().tolist())


def _cycle_set(cycles: list[list[list[torch.Tensor]]]) -> set[tuple[int, ...]]:
    keys = set()
    for order_cycles in cycles:
        for traj in order_cycles:
            for pt in traj:
                keys.add(_cycle_key(pt))
    return keys


def _count_all(cycles: list[list[list[torch.Tensor]]]) -> int:
    return sum(len(c) for c in cycles)


def _max_residual(
    A: torch.Tensor,
    W: torch.Tensor,
    h: torch.Tensor,
    cycles: list[list[list[torch.Tensor]]],
) -> float:
    worst = 0.0
    for oi, order_cycles in enumerate(cycles):
        order = oi + 1
        for traj in order_cycles:
            z = traj[0]
            for _ in range(order):
                z = A @ z + W @ torch.clamp(z, min=0.0) + h
            res = (z - traj[0]).norm().item()
            worst = max(worst, res)
    return worst


def _validate(
    A: torch.Tensor,
    W: torch.Tensor,
    h: torch.Tensor,
    cycles: list[list[list[torch.Tensor]]],
    atol: float = 1e-5,
) -> tuple[int, int]:
    n_valid = n_total = 0
    for oi, order_cycles in enumerate(cycles):
        order = oi + 1
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


def _gt_diff(found_set: set, gt_set: set) -> str:
    if not gt_set:
        return "—"
    extra = len(found_set - gt_set)
    missing = len(gt_set - found_set)
    if missing == 0 and extra == 0:
        return "0"
    parts = []
    if extra > 0:
        parts.append(f"+{extra}")
    if missing > 0:
        parts.append(f"-{missing}")
    return "".join(parts)


def _format_time(t_ms: float) -> str:
    if t_ms >= 1000:
        return f"{t_ms / 1000:.1f}s"
    return f"{t_ms:.0f}ms"


def _format_residual(res: float) -> str:
    if res == 0.0:
        return "0"
    return f"{res:.0e}"


def bench(fn, A, W, h, max_order, reps=3, **kw):
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


@dataclass
class VariantResult:
    name: str
    time_ms: float
    n_cycles: int
    n_valid: int
    n_total: int
    residual: float
    gt_diff: str
    skipped: bool = False


# ---------------------------------------------------------------------------
# Julia setup
# ---------------------------------------------------------------------------
def setup_julia():
    """Try to set up Julia. Returns callable or None."""
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

        def julia_find_cycles(A_np, W_np, h_np, max_order, outer=10, inner=20):
            result = jl.seval(
                "(A,W,h,mo,o,i)->begin; A_jl=Matrix{Float64}(A); W_jl=Matrix{Float64}(W); "
                "h_jl=Vector{Float64}(h); "
                "SCYFI.find_cycles(A_jl,W_jl,h_jl,mo,outer_loop_iterations=o,inner_loop_iterations=i); "
                "t = median([(@elapsed res=SCYFI.find_cycles(A_jl,W_jl,h_jl,mo,"
                "outer_loop_iterations=o,inner_loop_iterations=i)) for _ in 1:3])*1000; "
                "res = SCYFI.find_cycles(A_jl,W_jl,h_jl,mo,"
                "outer_loop_iterations=o,inner_loop_iterations=i); "
                "(t, res); end"
            )(A_np, W_np, h_np, max_order, outer, inner)

            t_ms = float(result[0])
            found_orders = result[1][0]
            all_cycles: list[list[list[torch.Tensor]]] = []
            for order_idx in range(len(found_orders)):
                order_cycles_jl = found_orders[order_idx]
                order_cycles: list[list[torch.Tensor]] = []
                for cycle_idx in range(len(order_cycles_jl)):
                    traj_jl = order_cycles_jl[cycle_idx]
                    traj: list[torch.Tensor] = []
                    for pt_idx in range(len(traj_jl)):
                        pt = torch.tensor(
                            np.array(traj_jl[pt_idx]), dtype=torch.float64
                        )
                        traj.append(pt)
                    order_cycles.append(traj)
                all_cycles.append(order_cycles)
            return t_ms, all_cycles

        return julia_find_cycles
    except Exception as e:
        print(f"Julia not available: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    julia_fn = setup_julia()

    print("Building embedded datasets...")
    t0 = time.perf_counter()
    datasets = build_all_datasets(target_dims=[5, 10, 20, 50], max_order=2)
    print(f"  {len(datasets)} datasets in {time.perf_counter() - t0:.1f}s\n")

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

    variants = [
        ("Original", find_cycles_original, {}),
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
        ("Newton", find_cycles_newton, {}),
        ("Hybrid", find_cycles_hybrid, {}),
        (
            "Schur",
            find_cycles_schur,
            {
                "B": 64,
                "outer_loop_iterations": 2,
                "inner_loop_iterations": 3,
            },
        ),
    ]

    variant_names = [v[0] for v in variants]
    if julia_fn:
        variant_names.append("Julia")

    exceeded: dict[tuple[str, int], bool] = {}

    # Collect all results
    all_results: list[tuple[str, int, int, int, list[VariantResult]]] = []

    for ds in datasets:
        gt_set = ds.gt_cycle_set()
        n_gt = ds.n_gt_cycles
        results: list[VariantResult] = []

        for vname, fn, kw in variants:
            key = (vname, ds.embedded_dim)
            if key in exceeded:
                results.append(VariantResult(vname, 0, 0, 0, 0, 0, "", skipped=True))
                continue

            torch.manual_seed(42)
            t_ms, result = bench(fn, ds.A, ds.W, ds.h, ds.max_order, **kw)

            if t_ms == float("inf"):
                exceeded[key] = True
                results.append(VariantResult(vname, 0, 0, 0, 0, 0, "", skipped=True))
                continue
            if t_ms > TIMEOUT * 1000:
                exceeded[key] = True

            cycles = result[0]
            n_c = _count_all(cycles)
            n_valid, n_total = _validate(ds.A, ds.W, ds.h, cycles)
            res = _max_residual(ds.A, ds.W, ds.h, cycles)
            found_set = _cycle_set(cycles)
            diff = _gt_diff(found_set, gt_set)

            results.append(VariantResult(vname, t_ms, n_c, n_valid, n_total, res, diff))

        # Julia
        if julia_fn:
            jl_key = ("Julia", ds.embedded_dim)
            if jl_key in exceeded:
                results.append(VariantResult("Julia", 0, 0, 0, 0, 0, "", skipped=True))
            else:
                try:
                    jl_t, jl_cycles = julia_fn(
                        ds.A.numpy(),
                        ds.W.numpy(),
                        ds.h.numpy(),
                        ds.max_order,
                    )
                    if jl_t > TIMEOUT * 1000:
                        exceeded[jl_key] = True
                    n_c = _count_all(jl_cycles)
                    n_valid, n_total = _validate(ds.A, ds.W, ds.h, jl_cycles)
                    res = _max_residual(ds.A, ds.W, ds.h, jl_cycles)
                    found_set = _cycle_set(jl_cycles)
                    diff = _gt_diff(found_set, gt_set)
                    results.append(
                        VariantResult("Julia", jl_t, n_c, n_valid, n_total, res, diff)
                    )
                except Exception as e:
                    results.append(
                        VariantResult(
                            "Julia", 0, 0, 0, 0, 0, f"ERR:{e!s:.8}", skipped=True
                        )
                    )

        all_results.append((ds.name, ds.embedded_dim, n_gt, ds.max_order, results))

    # --- Print markdown tables ---

    # Table 1: Timing
    print("## Timing (ms)\n")
    hdr = "| Dataset | Dim | " + " | ".join(variant_names) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(variant_names))) + "|"
    print(hdr)
    print(sep)
    for name, dim, n_gt, max_order, results in all_results:
        cells = []
        for r in results:
            if r.skipped:
                cells.append("SKIP")
            else:
                cells.append(_format_time(r.time_ms))
        row = f"| {name} | {dim} | " + " | ".join(cells) + " |"
        print(row)

    # Table 2: Cycles found
    print("\n## Cycles Found (#GT)\n")
    hdr = "| Dataset | Dim | GT | " + " | ".join(variant_names) + " |"
    sep = "|" + "|".join(["---"] * (3 + len(variant_names))) + "|"
    print(hdr)
    print(sep)
    for name, dim, n_gt, max_order, results in all_results:
        cells = []
        for r in results:
            if r.skipped:
                cells.append("SKIP")
            else:
                cells.append(str(r.n_cycles))
        row = f"| {name} | {dim} | {n_gt} | " + " | ".join(cells) + " |"
        print(row)

    # Table 3: GT diff (FP delta)
    print("\n## GT Diff (Δ)\n")
    print("0 = exact match, +N = extra found, -N = missed\n")
    hdr = "| Dataset | Dim | " + " | ".join(variant_names) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(variant_names))) + "|"
    print(hdr)
    print(sep)
    for name, dim, n_gt, max_order, results in all_results:
        cells = []
        for r in results:
            if r.skipped:
                cells.append("SKIP")
            else:
                cells.append(r.gt_diff)
        row = f"| {name} | {dim} | " + " | ".join(cells) + " |"
        print(row)

    # Table 4: Correctness
    print("\n## Correctness\n")
    print("✓ = all found cycles pass forward simulation, ✗N = N failures\n")
    hdr = "| Dataset | Dim | " + " | ".join(variant_names) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(variant_names))) + "|"
    print(hdr)
    print(sep)
    for name, dim, n_gt, max_order, results in all_results:
        cells = []
        for r in results:
            if r.skipped:
                cells.append("SKIP")
            elif r.n_valid == r.n_total:
                cells.append("✓")
            else:
                cells.append(f"✗{r.n_total - r.n_valid}")
        row = f"| {name} | {dim} | " + " | ".join(cells) + " |"
        print(row)

    # Table 5: Max Residual
    print("\n## Max Residual (||f^k(z) - z||)\n")
    hdr = "| Dataset | Dim | " + " | ".join(variant_names) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(variant_names))) + "|"
    print(hdr)
    print(sep)
    for name, dim, n_gt, max_order, results in all_results:
        cells = []
        for r in results:
            if r.skipped:
                cells.append("SKIP")
            else:
                cells.append(_format_residual(r.residual))
        row = f"| {name} | {dim} | " + " | ".join(cells) + " |"
        print(row)


if __name__ == "__main__":
    main()
