"""Fig 5: Homoclinic Chaos.

Uses PL map with Fig 5 parameters directly (no training).
Computes manifolds, detects homoclinic intersections,
generates bifurcation diagram and Lyapunov exponents.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic.analysis.homoclinic import (
    analytical_homoclinic_2d,
    find_homoclinic_intersections,
)
from dynamic.analysis.lyapunov import compute_lyapunov_exponents
from dynamic.analysis.manifolds import construct_manifold
from dynamic.analysis.pl_map_model import PLMapModel
from dynamic.analysis.scyfi import FixedPoint
from dynamic.analysis.subregions import classify_point
from dynamic.systems.pl_map import PLMap
from dynamic.viz.plotting import (
    plot_bifurcation,
    plot_lyapunov_spectrum,
    plot_state_space_2d,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def find_fixed_points_exhaustive(model) -> list[FixedPoint]:
    """Find all fixed points by exhaustive subregion search."""
    M = model.M
    fps = []
    for quad in range(2**M):
        bits = tuple((quad >> i) & 1 for i in range(M))
        D = torch.diag(torch.tensor(bits, dtype=torch.float32))
        J = model.A + model.W @ D
        I_minus_J = torch.eye(M) - J
        try:
            z_star = torch.linalg.solve(I_minus_J, model.h)
        except torch.linalg.LinAlgError:
            continue
        actual_bits = tuple(int(z_star[i] > 0) for i in range(M))
        if actual_bits != bits:
            continue
        evals, evecs = np.linalg.eig(J.detach().numpy())
        fps.append(
            FixedPoint(
                z=z_star.detach(),
                eigenvalues=evals,
                eigenvectors=evecs,
                classification=classify_point(evals),
                region_id=bits,
            )
        )
    return fps


def run_fig5():
    """Run the homoclinic chaos experiment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fig 5 PL map parameters
    pl_map = PLMap.fig5()
    model = PLMapModel(pl_map)

    print("Finding fixed points...")
    fps = find_fixed_points_exhaustive(model)
    print(f"  Found {len(fps)} fixed points")
    for fp in fps:
        print(f"    {fp.classification}: z={fp.z.numpy()}")

    saddles = [fp for fp in fps if fp.classification == "saddle"]

    if saddles:
        saddle = saddles[0]
        print("Computing manifolds...")
        stable = construct_manifold(model, saddle, sigma=+1, N_s=200, N_iter=20)
        unstable = construct_manifold(model, saddle, sigma=-1, N_s=200, N_iter=20)
        print(f"  Stable: {len(stable)} segments, Unstable: {len(unstable)} segments")

        # Detect homoclinic intersections using both methods
        print("Detecting homoclinic intersections...")
        # Method 1: analytical (Algorithm 4), best for 2D PL maps
        intersections = analytical_homoclinic_2d(
            model,
            saddle,
            N_s=500,
            N_iter=30,
        )
        print(f"  Analytical: {len(intersections)} intersection(s)")
        # Method 2: geometric segment intersection as supplement
        geom_intersections = find_homoclinic_intersections(
            stable,
            unstable,
            proximity_threshold=0.1,
        )
        # Merge, deduplicating
        for pt in geom_intersections:
            is_dup = any(torch.allclose(pt, ex, atol=1e-3) for ex in intersections)
            if not is_dup:
                intersections.append(pt)
        print(f"  Total: {len(intersections)} intersection(s)")

        # Plot state space with manifolds
        z0 = saddle.z + torch.tensor([0.01, 0.01])
        with torch.no_grad():
            traj = model.forward_trajectory(z0, T=500)

        fig = plot_state_space_2d(
            trajectories=[traj.numpy()],
            fixed_points=fps,
            manifolds=unstable,
            manifold_type="unstable",
            title=f"Fig 5: Homoclinic Chaos ({len(intersections)} intersections)",
        )
        fig.savefig(os.path.join(OUTPUT_DIR, "fig5_chaos_manifolds.png"), dpi=150)

    # Compute Lyapunov exponents
    print("Computing Lyapunov exponents...")
    z0 = torch.tensor([0.5, 0.3])
    exponents = compute_lyapunov_exponents(model, z0, T=5000)
    print(f"  Lyapunov exponents: {exponents}")
    is_chaotic = "CHAOTIC" if exponents[0] > 0 else "REGULAR"
    print(f"  Max exponent: {exponents[0]:.4f} ({is_chaotic})")

    # Bifurcation diagram: vary h1
    print("Computing bifurcation diagram...")
    h1_values = np.linspace(-1.0, 0.0, 50)
    all_attractor_vals = []

    for h1 in h1_values:
        with torch.no_grad():
            model.h[0] = h1
        z = torch.tensor([0.5, 0.3])
        # Transient
        with torch.no_grad():
            for _ in range(200):
                z = model.forward(z)
                if not torch.all(torch.isfinite(z)):
                    break
        # Collect attractor
        vals = []
        if torch.all(torch.isfinite(z)):
            with torch.no_grad():
                for _ in range(100):
                    z = model.forward(z)
                    if torch.all(torch.isfinite(z)):
                        vals.append(z[0].item())
                    else:
                        break
        all_attractor_vals.append(np.array(vals) if vals else np.array([np.nan]))

    fig_bif = plot_bifurcation(
        h1_values,
        all_attractor_vals,
        param_name="$h_1$",
        title="Fig 5C: Bifurcation Diagram",
    )
    fig_bif.savefig(os.path.join(OUTPUT_DIR, "fig5_bifurcation.png"), dpi=150)

    # Lyapunov spectrum vs h1
    print("Computing Lyapunov spectrum vs h1...")
    h1_lyap = np.linspace(-1.0, 0.0, 20)
    lyap_data = []
    for h1 in h1_lyap:
        with torch.no_grad():
            model.h[0] = h1
        z0 = torch.tensor([0.5, 0.3])
        exp = compute_lyapunov_exponents(model, z0, T=2000)
        lyap_data.append(exp)

    lyap_arr = np.array(lyap_data)
    fig_lyap = plot_lyapunov_spectrum(
        h1_lyap,
        lyap_arr,
        param_name="$h_1$",
        title="Fig 5D: Lyapunov Spectrum",
    )
    fig_lyap.savefig(os.path.join(OUTPUT_DIR, "fig5_lyapunov.png"), dpi=150)

    print(f"\nAll saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_fig5()
