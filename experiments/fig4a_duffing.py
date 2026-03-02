"""Fig 4A: Duffing Basin Boundaries.

Trains shPLRNN (M=2, H=10) on Duffing oscillator data.
Finds fixed points, computes stable manifold of saddle → basin boundary.
Target: Δ_σ ≈ 0.97 (Table 2, row 2).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic.analysis.manifolds import construct_manifold
from dynamic.analysis.quality import delta_sigma_statistic
from dynamic.analysis.subregions import classify_point, get_D
from dynamic.systems.duffing import generate_trajectory
from dynamic.training.configs import DUFFING_CONFIG
from dynamic.training.trainer import SparseTeacherForcingTrainer
from dynamic.viz.plotting import plot_basins_2d, plot_state_space_2d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_fig4a():
    """Run the Duffing basins experiment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate training data
    print("Generating Duffing oscillator data...")
    traj = generate_trajectory(x0=np.array([1.0, 0.0]), T=10.0, dt=0.01)
    data = torch.tensor(traj[:500], dtype=torch.float32)

    # Train model
    print("Training shPLRNN (M=2, H=10)...")
    torch.manual_seed(42)
    model = DUFFING_CONFIG.create_model()
    trainer = SparseTeacherForcingTrainer(model, DUFFING_CONFIG)
    losses = trainer.train(data, epochs=min(DUFFING_CONFIG.epochs, 2000))
    print(f"  Final loss: {losses[-1]:.6f}")

    # Find fixed points via exhaustive search
    print("Finding fixed points...")
    from dynamic.analysis.scyfi import FixedPoint

    all_fps = []
    M = model.M

    # For shPLRNN: iterate through latent space samples
    for _ in range(1000):
        z = torch.randn(M)
        with torch.no_grad():
            for _ in range(200):
                z_new = model.forward(z)
                if torch.allclose(z, z_new, atol=1e-6):
                    break
                z = z_new

        # Check if converged to FP
        with torch.no_grad():
            z_test = model.forward(z)
        if torch.allclose(z, z_test, atol=1e-5):
            J = model.get_jacobian(z).detach()
            evals = np.linalg.eig(J.numpy())[0]
            evecs = np.linalg.eig(J.numpy())[1]
            cls = classify_point(evals)
            # Deduplicate
            is_dup = any(
                torch.allclose(z, fp.z, atol=1e-3) for fp in all_fps
            )
            if not is_dup:
                all_fps.append(FixedPoint(
                    z=z.detach(),
                    eigenvalues=evals,
                    eigenvectors=evecs,
                    classification=cls,
                    region_id=model.get_subregion_id(z),
                ))

    print(f"  Found {len(all_fps)} fixed points:")
    for fp in all_fps:
        print(f"    {fp.classification}: z={fp.z.numpy()}")

    # Find saddle and compute manifold
    saddles = [fp for fp in all_fps if fp.classification == "saddle"]
    if saddles:
        saddle = saddles[0]
        print("Computing stable manifold of saddle...")
        stable = construct_manifold(model, saddle, sigma=+1, N_s=100, N_iter=15)

        delta_stat = delta_sigma_statistic(
            model, saddle.z, sigma=+1,
            U_min=saddle.z - 3.0, U_max=saddle.z + 3.0,
            N_samples=100, k_max=200,
        )
        print(f"  Δ_σ = {delta_stat:.4f}  (target ≈ 0.97)")

        fig = plot_state_space_2d(
            fixed_points=all_fps,
            manifolds=stable,
            manifold_type="stable",
            title=f"Fig 4A: Duffing Basins (Δ_σ = {delta_stat:.2f})",
        )
    else:
        print("  No saddle found. Plotting trajectories only.")
        fig = plot_state_space_2d(
            fixed_points=all_fps,
            title="Fig 4A: Duffing (no saddle found)",
        )

    fig.savefig(os.path.join(OUTPUT_DIR, "fig4a_duffing.png"), dpi=150)
    print(f"  Saved to {OUTPUT_DIR}/fig4a_duffing.png")


if __name__ == "__main__":
    run_fig4a()
