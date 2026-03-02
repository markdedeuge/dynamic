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
from dynamic.analysis.quality import delta_sigma, delta_sigma_statistic
from dynamic.analysis.scyfi import FixedPoint
from dynamic.analysis.subregions import classify_point, get_D
from dynamic.models.plrnn import PLRNN
from dynamic.systems.pl_map import PLMap
from dynamic.viz.plotting import plot_state_space_2d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def plmap_to_plrnn(pl_map: PLMap) -> PLRNN:
    """Convert a PLMap to a PLRNN model."""
    params = pl_map.to_plrnn_params()
    model = PLRNN(M=2)
    with torch.no_grad():
        A = params["A"]
        W = params["W"]
        h = params["h"]
        model.A.copy_(torch.diag(A) if A.dim() == 2 else A)
        model.W.copy_(W)
        model.h.copy_(h)
    return model


def find_saddle_points(model: PLRNN) -> list[FixedPoint]:
    """Find saddle fixed points by exhaustive subregion search."""
    M = model.A.shape[0]
    saddles = []

    for quad in range(2**M):
        bits = tuple((quad >> i) & 1 for i in range(M))
        D = torch.diag(torch.tensor(bits, dtype=torch.float32))
        J = torch.diag(model.A) + model.W @ D
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
    model = plmap_to_plrnn(pl_map)

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


if __name__ == "__main__":
    run_fig3()
