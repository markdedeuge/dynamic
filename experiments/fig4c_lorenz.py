"""Fig 4C: Lorenz-63 Manifolds.

Trains shPLRNN (M=3, H=20) on Lorenz data.
Computes stable and unstable manifolds.
Target: Δ_σ ≈ 0.78 (Table 2, row 4).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic.analysis.manifolds import construct_manifold
from dynamic.analysis.quality import delta_sigma_statistic
from dynamic.analysis.scyfi import FixedPoint
from dynamic.analysis.subregions import classify_point
from dynamic.systems.lorenz63 import generate_trajectory
from dynamic.training.configs import LORENZ_FIG4C_CONFIG
from dynamic.training.trainer import SparseTeacherForcingTrainer
from dynamic.viz.plotting import plot_state_space_3d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_fig4c():
    """Run the Lorenz-63 manifolds experiment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate training data
    print("Generating Lorenz-63 data...")
    traj = generate_trajectory(x0=np.array([1.0, 1.0, 1.0]), T=50.0, dt=0.01)
    data = torch.tensor(traj[:1000], dtype=torch.float32)

    # Train model
    print(f"Training shPLRNN (M=3, H={LORENZ_FIG4C_CONFIG.H})...")
    torch.manual_seed(42)
    model = LORENZ_FIG4C_CONFIG.create_model()
    epochs = min(LORENZ_FIG4C_CONFIG.epochs, 500)
    trainer = SparseTeacherForcingTrainer(model, LORENZ_FIG4C_CONFIG)
    losses = trainer.train(data, epochs=epochs)
    print(f"  Final loss: {losses[-1]:.6f}")

    # Find fixed points
    print("Finding fixed points...")
    all_fps = []
    for _ in range(500):
        z = torch.randn(3) * 2
        with torch.no_grad():
            for _ in range(300):
                z_new = model.forward(z)
                if torch.allclose(z, z_new, atol=1e-6):
                    break
                z = z_new
                if not torch.all(torch.isfinite(z)):
                    break

        if not torch.all(torch.isfinite(z)):
            continue

        with torch.no_grad():
            z_test = model.forward(z)
        if torch.allclose(z, z_test, atol=1e-4):
            J = model.get_jacobian(z).detach()
            evals = np.linalg.eig(J.numpy())[0]
            evecs = np.linalg.eig(J.numpy())[1]
            cls = classify_point(evals)
            is_dup = any(torch.allclose(z, fp.z, atol=1e-2) for fp in all_fps)
            if not is_dup:
                all_fps.append(FixedPoint(
                    z=z.detach(), eigenvalues=evals, eigenvectors=evecs,
                    classification=cls, region_id=model.get_subregion_id(z),
                ))

    print(f"  Found {len(all_fps)} fixed points")
    for fp in all_fps:
        print(f"    {fp.classification}: z={fp.z.numpy()}")

    saddles = [fp for fp in all_fps if fp.classification == "saddle"]
    if saddles:
        saddle = saddles[0]
        print("Computing manifolds...")
        stable = construct_manifold(model, saddle, sigma=+1, N_s=100, N_iter=10)
        unstable = construct_manifold(model, saddle, sigma=-1, N_s=100, N_iter=10)

        delta_stat = delta_sigma_statistic(
            model, saddle.z, sigma=-1,
            U_min=saddle.z - 5.0, U_max=saddle.z + 5.0,
            N_samples=50, k_max=100,
        )
        print(f"  Δ_σ = {delta_stat:.4f}  (target ≈ 0.78)")

        # Plot trajectory + manifolds
        with torch.no_grad():
            learned_traj = model.forward_trajectory(data[0], T=500)
        fig = plot_state_space_3d(
            trajectories=[learned_traj.numpy()],
            fixed_points=all_fps,
            manifolds=stable + unstable,
            title=f"Fig 4C: Lorenz-63 (Δ_σ = {delta_stat:.2f})",
        )
    else:
        print("  No saddle found.")
        fig = plot_state_space_3d(
            fixed_points=all_fps,
            title="Fig 4C: Lorenz-63 (no saddle)",
        )

    fig.savefig(os.path.join(OUTPUT_DIR, "fig4c_lorenz.png"), dpi=150)
    print(f"  Saved to {OUTPUT_DIR}/fig4c_lorenz.png")


if __name__ == "__main__":
    run_fig4c()
