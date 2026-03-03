"""Fig 3: Toy PL Map Validation.

No training required — uses PLMap directly as a PLRNN.
Computes stable/unstable manifolds of saddle fixed points.
Validates with Δ_σ ≈ 0.98 (Table 2, row 1).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic.analysis.manifolds import construct_manifold
from dynamic.analysis.pl_map_model import PLMapModel
from dynamic.analysis.quality import delta_sigma_statistic
from dynamic.analysis.scyfi import FixedPoint
from dynamic.analysis.subregions import classify_point
from dynamic.systems.pl_map import PLMap
from dynamic.viz.plotting import plot_state_space_2d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def find_saddle_points(model) -> list[FixedPoint]:
    """Find saddle fixed points by exhaustive subregion search."""
    M = model.M
    saddles = []

    for quad in range(2**M):
        bits = tuple((quad >> i) & 1 for i in range(M))
        D = torch.diag(torch.tensor(bits, dtype=torch.float32))
        J = model.A + model.W @ D
        I_minus_J = torch.eye(M) - J

        try:
            z_star = torch.linalg.solve(I_minus_J, model.h)
        except torch.linalg.LinAlgError:
            continue

        # Check self-consistency: region of z* matches assumed D
        actual_bits = tuple(int(z_star[i] > 0) for i in range(M))
        if actual_bits != bits:
            continue

        evals, evecs = np.linalg.eig(J.detach().numpy())
        classification = classify_point(evals)

        if classification == "saddle":
            saddles.append(FixedPoint(
                z=z_star.detach(),
                eigenvalues=evals,
                eigenvectors=evecs,
                classification=classification,
                region_id=bits,
            ))

    return saddles


def run_fig3():
    """Run the Fig 3 experiment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fig 3A left — simple saddle manifolds
    pl_map = PLMap.fig3a_left()
    model = PLMapModel(pl_map)

    print("Finding saddle points...")
    saddles = find_saddle_points(model)
    print(f"  Found {len(saddles)} saddle(s)")

    if not saddles:
        print("  No saddles found, skipping manifold computation.")
        return

    saddle = saddles[0]
    print(f"  Saddle at z* = {saddle.z.numpy()}")
    print(f"  Eigenvalues: {saddle.eigenvalues}")

    # Compute manifolds
    print("Computing stable manifold...")
    stable = construct_manifold(model, saddle, sigma=+1, N_s=200, N_iter=20)
    print(f"  {len(stable)} segments")

    print("Computing unstable manifold...")
    unstable = construct_manifold(model, saddle, sigma=-1, N_s=200, N_iter=20)
    print(f"  {len(unstable)} segments")

    # Compute Δ_σ
    print("Computing Δ_σ...")
    delta_stat = delta_sigma_statistic(
        model, saddle.z, sigma=-1,
        U_min=saddle.z - 2.0,
        U_max=saddle.z + 2.0,
        N_samples=200, k_max=200,
    )
    print(f"  Δ_σ = {delta_stat:.4f}  (target ≈ 0.98)")

    # Generate trajectory for plot context
    z0 = saddle.z + torch.tensor([0.1, 0.1])
    with torch.no_grad():
        traj = model.forward_trajectory(z0, T=200)
    trajectories = [traj.numpy()]

    # Plot
    fig = plot_state_space_2d(
        trajectories=trajectories,
        fixed_points=[saddle],
        manifolds=unstable,
        manifold_type="unstable",
        title=f"Fig 3A: Toy PL Map (Δ_σ = {delta_stat:.2f})",
    )
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_toy_validation.png"), dpi=150)
    print(f"  Saved to {OUTPUT_DIR}/fig3_toy_validation.png")

    return delta_stat


def find_all_fixed_points(model) -> list[FixedPoint]:
    """Find all fixed points (stable, unstable, saddle) exhaustively."""
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
        fps.append(FixedPoint(
            z=z_star.detach(),
            eigenvalues=evals,
            eigenvectors=evecs,
            classification=classify_point(evals),
            region_id=bits,
        ))

    return fps


def run_fig3b():
    """Run Fig 3B: period-3/4 cycle basins of attraction.

    Uses PLMap.fig3b_left (period-4) and PLMap.fig3b_right (period-3).
    Finds fixed points and saddles, computes stable manifolds to
    delineate basins, and generates basin plots.
    """
    from dynamic.viz.plotting import plot_basins_2d

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = [
        ("fig3b_left (period-4)", PLMap.fig3b_left()),
        ("fig3b_right (period-3)", PLMap.fig3b_right()),
    ]

    for name, pl_map in configs:
        print(f"\n=== {name} ===")
        model = PLMapModel(pl_map)

        # Find all fixed points
        all_fps = find_all_fixed_points(model)
        print(f"  Found {len(all_fps)} fixed points:")
        for fp in all_fps:
            print(
                f"    {fp.classification}:"
                f" z={fp.z.numpy()}, λ={fp.eigenvalues}"
            )

        # Generate trajectories from various initial conditions
        trajs = []
        for ic in [
            torch.tensor([0.5, 0.5]),
            torch.tensor([-0.5, 0.3]),
            torch.tensor([0.3, -0.5]),
            torch.tensor([-0.3, -0.3]),
        ]:
            traj = model.forward_trajectory(ic, T=200)
            trajs.append(traj.numpy())

        # Compute stable manifolds of saddles
        saddles = [
            fp for fp in all_fps if fp.classification == "saddle"
        ]
        all_manifolds = []
        for saddle in saddles:
            stable = construct_manifold(
                model, saddle, sigma=+1, N_s=100, N_iter=10,
            )
            all_manifolds.extend(stable)

        # State space plot with trajectories and manifolds
        fig = plot_state_space_2d(
            trajectories=trajs,
            fixed_points=all_fps,
            manifolds=all_manifolds,
            manifold_type="stable",
            title=f"Fig 3B: {name}",
        )
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"fig3b_{safe_name}.png"),
            dpi=150,
        )
        print("  Saved state space plot")

        # Basin plot
        stable_fps = [
            fp for fp in all_fps if fp.classification == "stable"
        ]
        if stable_fps:
            fig_basin = plot_basins_2d(
                model, stable_fps,
                x_range=(-2.0, 2.0), y_range=(-2.0, 2.0),
                resolution=80,
                title=f"Fig 3B Basins: {name}",
            )
            fig_basin.savefig(
                os.path.join(
                    OUTPUT_DIR, f"fig3b_basins_{safe_name}.png",
                ),
                dpi=150,
            )
            print("  Saved basin plot")
        else:
            print("  No stable FPs found — skipping basin plot")

    print(f"\nAll saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_fig3()
    run_fig3b()
