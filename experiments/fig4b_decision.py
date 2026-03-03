"""Fig 4B: Decision-Making Multistability.

Trains ALRNN (M=15, P=6) on decision-making task data.
Multi-seed training: trains N_SEEDS models, picks the one with the best
saddle for manifold analysis.
Target: Δ_σ ≈ 0.95 (Table 2, row 3).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic.analysis.manifolds import construct_manifold
from dynamic.analysis.quality import delta_sigma_statistic
from dynamic.analysis.scyfi import FixedPoint
from dynamic.analysis.scyfi_vectorised import find_cycles_vectorised
from dynamic.analysis.subregions import classify_point
from dynamic.systems.decision import generate_trajectory
from dynamic.training.configs import DECISION_CONFIG
from dynamic.training.trainer import SparseTeacherForcingTrainer
from dynamic.viz.plotting import plot_state_space_3d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_SEEDS = 10


def _scyfi_fps_from_alrnn(model) -> list[FixedPoint]:
    """Use VecC SCYFI to find fixed points in an ALRNN."""
    all_cycles, all_eigvals = find_cycles_vectorised(
        torch.diag(model.A.detach()),
        model.W.detach(),
        model.h.detach(),
        max_order=1,
        outer_loop_iterations=200,
        inner_loop_iterations=500,
        batch_size=64,
        compiled=True,
    )

    fps = []
    for order_idx in range(len(all_cycles)):
        for cycle_idx, traj in enumerate(all_cycles[order_idx]):
            z = traj[0].float()
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
    """Score how well-separated the saddle eigenvalues are.

    Requires both a saddle AND at least one stable FP for valid Δ_σ.
    """
    saddles = [fp for fp in fps if fp.classification == "saddle"]
    stables = [fp for fp in fps if fp.classification == "stable"]
    if not saddles or not stables:
        return 0.0

    best = 0.0
    for s in saddles:
        evals_abs = np.abs(s.eigenvalues)
        separation = np.sum(np.abs(evals_abs - 1.0))
        best = max(best, separation)
    return best


def run_fig4b():
    """Run the decision-making multistability experiment with multi-seed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate training data from two different ICs (two choices)
    print("Generating decision-making data...")
    traj1 = generate_trajectory(x0=np.array([0.1, 0.0, 0.0]), T=20.0, dt=0.01)
    traj2 = generate_trajectory(x0=np.array([0.0, 0.1, 0.0]), T=20.0, dt=0.01)
    data_np = np.concatenate([traj1[:500], traj2[:500]], axis=0)

    M = DECISION_CONFIG.M
    data = torch.zeros(data_np.shape[0], M, dtype=torch.float32)
    data[:, :3] = torch.tensor(data_np, dtype=torch.float32)

    # Multi-seed training
    print(f"Multi-seed training (N={N_SEEDS} seeds)...")
    best_model = None
    best_fps: list[FixedPoint] = []
    best_score = -1.0
    best_loss = float("inf")

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        model = DECISION_CONFIG.create_model()
        trainer = SparseTeacherForcingTrainer(model, DECISION_CONFIG)
        losses = trainer.train(data, epochs=DECISION_CONFIG.epochs)
        loss = losses[-1]

        fps = _scyfi_fps_from_alrnn(model)
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
        print(f"    {fp.classification}")

    saddles = [fp for fp in all_fps if fp.classification == "saddle"]
    if saddles:
        saddle = max(
            saddles,
            key=lambda s: np.sum(np.abs(np.abs(s.eigenvalues) - 1.0)),
        )
        print("Computing stable manifold...")
        stable = construct_manifold(model, saddle, sigma=+1, N_s=50, N_iter=10)

        # Extract manifold points for Δ_σ threshold
        manifold_pts_list = [
            seg.points
            for seg in stable
            if seg.points.numel() > 0 and seg.points.dim() > 1
        ]
        if manifold_pts_list:
            manifold_pts = torch.cat(manifold_pts_list)
        else:
            manifold_pts = None
        n_mpts = len(manifold_pts) if manifold_pts is not None else 0
        print(f"  Manifold points: {n_mpts}")

        delta_stat = delta_sigma_statistic(
            model,
            saddle.z,
            sigma=+1,
            U_min=saddle.z - 2.0,
            U_max=saddle.z + 2.0,
            N_samples=100,
            k_max=200,
            manifold_points=manifold_pts,
        )
        print(f"  Δ_σ = {delta_stat:.4f}  (target ≈ 0.95)")

        fig = plot_state_space_3d(
            fixed_points=all_fps,
            manifolds=stable,
            dims=(0, 1, 2),
            title=f"Fig 4B: Decision-Making (Δ_σ = {delta_stat:.2f})",
        )
    else:
        print("  No saddle found across all seeds.")
        fig = plot_state_space_3d(
            fixed_points=all_fps,
            dims=(0, 1, 2),
            title="Fig 4B: Decision-Making (no saddle)",
        )

    fig.savefig(os.path.join(OUTPUT_DIR, "fig4b_decision.png"), dpi=150)
    print(f"  Saved to {OUTPUT_DIR}/fig4b_decision.png")


if __name__ == "__main__":
    run_fig4b()
