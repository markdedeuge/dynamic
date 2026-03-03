"""4-way benchmark: reference vs fast vs fused vs Julia."""
# ruff: noqa: E501

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def setup_julia():
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


def julia_timed(jl, A_np, W_np, h_np, max_order, outer, inner):
    t_ms = jl.seval("""
        (A, W, h, max_order, outer, inner) -> begin
            A_jl = Matrix{Float64}(A)
            W_jl = Matrix{Float64}(W)
            h_jl = Vector{Float64}(h)
            SCYFI.find_cycles(A_jl, W_jl, h_jl, max_order,
                outer_loop_iterations=outer, inner_loop_iterations=inner)
            t = median([
                (@elapsed SCYFI.find_cycles(A_jl, W_jl, h_jl, max_order,
                    outer_loop_iterations=outer, inner_loop_iterations=inner))
                for _ in 1:5
            ])
            return t * 1000
        end
    """)(A_np, W_np, h_np, max_order, outer, inner)
    return float(t_ms)


def bench(fn, A, W, h, order, outer, inner, **kw):
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        fn(A, W, h, order, outer_loop_iterations=outer,
           inner_loop_iterations=inner, **kw)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    from dynamic.analysis.scyfi import find_cycles
    from dynamic.analysis.scyfi_fast import find_cycles_fast
    from dynamic.analysis.scyfi_fused import find_cycles_fused

    outer, inner = 10, 20

    print("Loading Julia...")
    jl = setup_julia()
    print("Julia ready.\n")

    header = (
        f"{'Dim':>4} {'Ord':>4} "
        f"{'Reference':>11} {'Fast':>11} {'Fused':>11} {'Julia':>11}  "
        f"{'Fast/Ref':>9} {'Fused/Ref':>10} {'Fused/Jl':>10}"
    )
    print(header)
    print("-" * 100)

    for dim in [2, 5, 10, 20]:
        for order in [1, 4, 8]:
            torch.manual_seed(42)
            A = torch.diag(torch.randn(dim, dtype=torch.float64))
            W = torch.randn(dim, dim, dtype=torch.float64) * 0.3
            W.fill_diagonal_(0.0)
            h = torch.randn(dim, dtype=torch.float64)

            ref_t = bench(find_cycles, A, W, h, order, outer, inner)
            fast_t = bench(find_cycles_fast, A, W, h, order, outer, inner)
            fused_t = bench(find_cycles_fused, A, W, h, order, outer, inner)
            jl_t = julia_timed(
                jl, A.numpy(), W.numpy(), h.numpy(), order, outer, inner
            )

            print(
                f"{dim:4d} {order:4d} "
                f"{ref_t:9.1f}ms {fast_t:9.1f}ms "
                f"{fused_t:9.1f}ms {jl_t:9.1f}ms  "
                f"{ref_t/fast_t:7.2f}x  "
                f"{ref_t/fused_t:8.2f}x  "
                f"{fused_t/jl_t:8.2f}x"
            )


if __name__ == "__main__":
    main()
