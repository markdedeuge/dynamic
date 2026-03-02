"""Fig 4B: Decision-Making Multistability.

Trains ALRNN (M=15, P=6) on decision-making task data.
Finds two stable FPs and saddle, computes stable manifold.
Target: Δ_σ ≈ 0.95 (Table 2, row 3).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic.analysis.manifolds import construct_manifold
from dynamic.analysis.quality import delta_sigma_statistic
from dynamic.analysis.scyfi import FixedPoint
from dynamic.analysis.subregions import classify_point
from dynamic.systems.decision import generate_trajectory
from dynamic.training.configs import DECISION_CONFIG
from dynamic.training.trainer import SparseTeacherForcingTrainer
from dynamic.viz.plotting import plot_state_space_3d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_fig4b():
    """Run the decision-making multistability experiment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate training data
    print("Generating decision-making data...")
    traj1 = generate_trajectory(x0=np.array([0.1, 0.0, 0.0]), T=20.0, dt=0.01)
    traj2 = generate_trajectory(x0=np.array([0.0, 0.1, 0.0]), T=20.0, dt=0.01)
    data_np = np.concatenate([traj1[:500], traj2[:500]], axis=0)

    M = DECISION_CONFIG.M
    data = torch.zeros(data_np.shape[0], M, dtype=torch.float32)
    data[:, :3] = torch.tensor(data_np, dtype=torch.float32)

    # Train model
    print(f"Training ALRNN (M={M}, P={DECISION_CONFIG.P})...")
    torch.manual_seed(42)
    model = DECISION_CONFIG.create_model()
    epochs = min(DECISION_CONFIG.epochs, 2000)
    trainer = SparseTeacherForcingTrainer(model, DECISION_CONFIG)
    losses = trainer.train(data, epochs=epochs)
    print(f"  Final loss: {losses[-1]:.6f}")

    # Find fixed points via iteration
    print("Finding fixed points...")
    all_fps = []
    for _ in range(500):
        z = torch.randn(M) * 0.5
        with torch.no_grad():
            for _ in range(500):
                z_new = model.forward(z)
                if torch.allclose(z, z_new, atol=1e-6):
                    break
                z = z_new

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
        print(f"    {fp.classification}")

    # Find saddle
    saddles = [fp for fp in all_fps if fp.classification == "saddle"]
    if saddles:
        saddle = saddles[0]
        print("Computing stable manifold...")
        stable = construct_manifold(model, saddle, sigma=+1, N_s=50, N_iter=10)

        delta_stat = delta_sigma_statistic(
            model, saddle.z, sigma=+1,
            U_min=saddle.z - 2.0, U_max=saddle.z + 2.0,
            N_samples=50, k_max=100,
        )
        print(f"  Δ_σ = {delta_stat:.4f}  (target ≈ 0.95)")

        fig = plot_state_space_3d(
            fixed_points=all_fps,
            manifolds=stable,
            dims=(0, 1, 2),
            title=f"Fig 4B: Decision-Making (Δ_σ = {delta_stat:.2f})",
        )
    else:
        print("  No saddle found.")
        fig = plot_state_space_3d(
            fixed_points=all_fps,
            dims=(0, 1, 2),
            title="Fig 4B: Decision-Making (no saddle)",
        )

    fig.savefig(os.path.join(OUTPUT_DIR, "fig4b_decision.png"), dpi=150)
    print(f"  Saved to {OUTPUT_DIR}/fig4b_decision.png")


if __name__ == "__main__":
    run_fig4b()
