"""5-way benchmark: reference vs fused vs vectorised vs vectorised+compiled vs Julia."""
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
    xval_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "cross_validation")
    scyfi_src = os.path.abspath(os.path.join(xval_dir, "..", "..", "reference", "SCYFI", "src"))
    bifur_stripped = os.path.abspath(os.path.join(xval_dir, "helpers_bifurcation_stripped.jl"))
    bootstrap = os.path.abspath(os.path.join(xval_dir, "scyfi_bootstrap.jl"))
    jl.seval('import Pkg; try; using Distributions; catch; Pkg.add("Distributions"); using Distributions; end')
    jl.seval(f"""
        module SCYFI
            using LinearAlgebra; using Random; using Distributions
            include("{scyfi_src}/utilities/helpers.jl")
            include("{scyfi_src}/utilities/exhaustive_search.jl")
            include("{bifur_stripped}")
            include("{bootstrap}")
        end
    """)
    jl.seval("using LinearAlgebra, Statistics")
    return jl


def julia_timed(jl, A_np, W_np, h_np, max_order, outer, inner):
    return float(jl.seval("""
        (A, W, h, mo, o, i) -> begin
            A_jl, W_jl, h_jl = Matrix{Float64}(A), Matrix{Float64}(W), Vector{Float64}(h)
            SCYFI.find_cycles(A_jl, W_jl, h_jl, mo, outer_loop_iterations=o, inner_loop_iterations=i)
            median([(@elapsed SCYFI.find_cycles(A_jl, W_jl, h_jl, mo, outer_loop_iterations=o, inner_loop_iterations=i)) for _ in 1:5]) * 1000
        end
    """)(A_np, W_np, h_np, max_order, outer, inner))


def bench(fn, A, W, h, order, outer, inner, **kw):
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        fn(A, W, h, order, outer_loop_iterations=outer, inner_loop_iterations=inner, **kw)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    from dynamic.analysis.scyfi import find_cycles
    from dynamic.analysis.scyfi_fused import find_cycles_fused
    from dynamic.analysis.scyfi_vectorised import find_cycles_vectorised

    outer, inner = 10, 20

    print("Loading Julia...")
    jl = setup_julia()
    print("Julia ready.\n")

    # Warmup compiled kernels
    print("Warming up torch.compile...")
    A_w = torch.diag(torch.randn(2, dtype=torch.float64))
    W_w = torch.randn(2, 2, dtype=torch.float64)
    h_w = torch.randn(2, dtype=torch.float64)
    find_cycles_vectorised(A_w, W_w, h_w, 1, outer_loop_iterations=2, inner_loop_iterations=2, batch_size=4, compiled=True)
    print("Warmup done.\n")

    header = (
        f"{'Dim':>4} {'Ord':>4} "
        f"{'Ref':>9} {'Fused':>9} {'Vec':>9} {'Vec+C':>9} {'Julia':>9}  "
        f"{'Vec/Ref':>8} {'Vec+C/Ref':>10} {'Vec/Jl':>8} {'Vec+C/Jl':>9}"
    )
    print(header)
    print("-" * 115)

    for dim in [2, 5, 10, 20]:
        for order in [1, 4, 8]:
            torch.manual_seed(42)
            A = torch.diag(torch.randn(dim, dtype=torch.float64))
            W = torch.randn(dim, dim, dtype=torch.float64) * 0.3
            W.fill_diagonal_(0.0)
            h = torch.randn(dim, dtype=torch.float64)

            ref_t = bench(find_cycles, A, W, h, order, outer, inner)
            fused_t = bench(find_cycles_fused, A, W, h, order, outer, inner)
            vec_t = bench(find_cycles_vectorised, A, W, h, order, outer, inner, batch_size=64)
            vecc_t = bench(find_cycles_vectorised, A, W, h, order, outer, inner, batch_size=64, compiled=True)
            jl_t = julia_timed(jl, A.numpy(), W.numpy(), h.numpy(), order, outer, inner)

            print(
                f"{dim:4d} {order:4d} "
                f"{ref_t:7.1f}ms {fused_t:7.1f}ms {vec_t:7.1f}ms {vecc_t:7.1f}ms {jl_t:7.1f}ms  "
                f"{ref_t/vec_t:6.2f}x  {ref_t/vecc_t:8.2f}x  "
                f"{vec_t/jl_t:6.2f}x  {vecc_t/jl_t:7.2f}x"
            )


if __name__ == "__main__":
    main()
