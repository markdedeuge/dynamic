"""Fig 4C: Lorenz-63 Manifolds.

Trains shPLRNN (M=3, H=20) on Lorenz data.
Multi-seed training: trains N_SEEDS models, picks the one with the best
saddle for manifold analysis.
Target: Δ_σ ≈ 0.78 (Table 2, row 4).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic.analysis.manifolds import construct_manifold
from dynamic.analysis.quality import delta_sigma_statistic
from dynamic.analysis.scyfi import FixedPoint
from dynamic.analysis.scyfi_vectorised import find_cycles_sh_vectorised
from dynamic.analysis.subregions import classify_point
from dynamic.systems.lorenz63 import generate_trajectory
from dynamic.training.configs import LORENZ_FIG4C_CONFIG
from dynamic.training.trainer import SparseTeacherForcingTrainer
from dynamic.viz.plotting import plot_state_space_3d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_SEEDS = 10


def _scyfi_fps_from_shplrnn(model, max_order: int = 1) -> list[FixedPoint]:
    """Use SCYFI to find fixed points/cycles in a shPLRNN."""
    with torch.no_grad():
        A = model.A.detach()
        W1 = model.W1.detach()
        W2 = model.W2.detach()
        h1 = model.h1.detach()
        h2 = model.h2.detach()

    all_cycles, all_eigvals = find_cycles_sh_vectorised(
        A,
        W1,
        W2,
        h1,
        h2,
        max_order,
        outer_loop_iterations=100,
        inner_loop_iterations=500,
        batch_size=64,
    )

    fps = []
    for order_idx in range(len(all_cycles)):
        for cycle_idx, traj in enumerate(all_cycles[order_idx]):
            z = traj[0]
            evals = all_eigvals[order_idx][cycle_idx]
            cls = classify_point(evals)

            J = model.get_jacobian(z).detach().numpy()
            evecs = np.linalg.eig(J)[1]

            fps.append(
                FixedPoint(
                    z=z.detach(),
                    eigenvalues=evals,
                    eigenvectors=evecs,
                    classification=cls,
                    region_id=model.get_subregion_id(z),
                )
            )

    return fps


def _saddle_quality(fps: list[FixedPoint]) -> float:
    """Score how well-separated the saddle eigenvalues are."""
    saddles = [fp for fp in fps if fp.classification == "saddle"]
    if not saddles:
        return 0.0

    best = 0.0
    for s in saddles:
        evals_abs = np.abs(s.eigenvalues)
        separation = np.sum(np.abs(evals_abs - 1.0))
        best = max(best, separation)
    return best


def run_fig4c():
    """Run the Lorenz-63 manifolds experiment with multi-seed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate training data — long chaotic trajectory
    print("Generating Lorenz-63 data...")
    traj = generate_trajectory(x0=np.array([1.0, 1.0, 1.0]), T=100.0, dt=0.01)
    data = torch.tensor(traj[:5000], dtype=torch.float32)

    # Multi-seed training
    print(f"Multi-seed training (N={N_SEEDS} seeds)...")
    best_model = None
    best_fps: list[FixedPoint] = []
    best_score = -1.0
    best_loss = float("inf")

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        model = LORENZ_FIG4C_CONFIG.create_model()
        trainer = SparseTeacherForcingTrainer(model, LORENZ_FIG4C_CONFIG)
        losses = trainer.train(data, epochs=LORENZ_FIG4C_CONFIG.epochs)
        loss = losses[-1]

        fps = _scyfi_fps_from_shplrnn(model, max_order=1)
        score = _saddle_quality(fps)
        n_saddles = sum(1 for fp in fps if fp.classification == "saddle")

        print(
            f"  Seed {seed}: loss={loss:.6f}, "
            f"FPs={len(fps)} ({n_saddles} saddles), "
            f"quality={score:.4f}"
        )

        if score > best_score or (score == best_score and loss < best_loss):
            best_model = model
            best_fps = fps
            best_score = score
            best_loss = loss

    print(f"\nBest seed: quality={best_score:.4f}, loss={best_loss:.6f}")

    all_fps = best_fps
    model = best_model

    print(f"  Found {len(all_fps)} fixed points:")
    for fp in all_fps:
        print(f"    {fp.classification}: z={fp.z.numpy()}")

    saddles = [fp for fp in all_fps if fp.classification == "saddle"]
    if saddles:
        saddle = max(
            saddles,
            key=lambda s: np.sum(np.abs(np.abs(s.eigenvalues) - 1.0)),
        )
        print("Computing manifolds...")
        stable = construct_manifold(model, saddle, sigma=+1, N_s=100, N_iter=10)
        unstable = construct_manifold(
            model,
            saddle,
            sigma=-1,
            N_s=100,
            N_iter=10,
        )

        delta_stat = delta_sigma_statistic(
            model,
            saddle.z,
            sigma=-1,
            U_min=saddle.z - 5.0,
            U_max=saddle.z + 5.0,
            N_samples=100,
            k_max=200,
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
        print("  No saddle found across all seeds.")
        fig = plot_state_space_3d(
            fixed_points=all_fps,
            title="Fig 4C: Lorenz-63 (no saddle)",
        )

    fig.savefig(os.path.join(OUTPUT_DIR, "fig4c_lorenz.png"), dpi=150)
    print(f"  Saved to {OUTPUT_DIR}/fig4c_lorenz.png")


if __name__ == "__main__":
    run_fig4c()
