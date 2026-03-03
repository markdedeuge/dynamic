"""Fig 2: Invertibility Regularization Ablation.

Trains ALRNN on Lorenz-63 with varying M and λ_invert.
Measures proportion of invertible subregions and D_stsp
reconstruction quality.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic.systems.lorenz63 import generate_trajectory
from dynamic.training.configs import TrainingConfig
from dynamic.training.trainer import SparseTeacherForcingTrainer
from dynamic.viz.plotting import plot_invertibility

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def measure_invertibility(model, z_samples: torch.Tensor) -> float:
    """Proportion of sampled subregions with det(J) > 0."""
    count_pos = 0
    total = 0
    with torch.no_grad():
        for i in range(len(z_samples)):
            J = model.get_jacobian(z_samples[i])
            det = torch.det(J).item()
            if det > 0:
                count_pos += 1
            total += 1
    return count_pos / max(total, 1)


def compute_d_stsp(
    model, data: torch.Tensor, n_steps: int = 50,
) -> float:
    """D_stsp: state-space reconstruction distance.

    Measures how well a free-running model trajectory matches the
    original data, averaged over sliding windows.

    Returns mean L2 distance between free-running and ground truth.
    """
    M = data.shape[1]
    distances = []
    step_size = max(1, len(data) // 20)

    with torch.no_grad():
        for start in range(0, len(data) - n_steps, step_size):
            z = data[start].clone()
            total_dist = 0.0
            for t in range(1, n_steps):
                z = model.forward(z)
                if not torch.all(torch.isfinite(z)):
                    break
                gt = data[start + t]
                total_dist += torch.norm(
                    z[:M] - gt[:M],
                ).item()
            distances.append(total_dist / n_steps)

    return float(np.mean(distances)) if distances else float("inf")


def run_fig2():
    """Run the invertibility ablation experiment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate Lorenz data
    print("Generating Lorenz-63 data...")
    traj = generate_trajectory(
        x0=np.array([1.0, 1.0, 1.0]), T=20.0, dt=0.01,
    )
    data = torch.tensor(traj[:500], dtype=torch.float32)

    dimensions = [10, 20, 30, 50]
    prop_no_reg = []
    prop_with_reg = []
    d_stsp_no = []
    d_stsp_reg = []

    for M in dimensions:
        P = max(3, M // 3)
        print(f"\n--- M={M}, P={P} ---")

        # Pad data to match M dimensions
        data_padded = torch.zeros(data.shape[0], M)
        data_padded[:, :3] = data

        # Without regularization
        torch.manual_seed(42)
        config_no = TrainingConfig(
            model_type="alrnn", M=M, H=None, P=P,
            sequence_length=50, noise_std=0.0,
            lambda_invert=0.0,
            batch_size=4, epochs=100,
            learning_rate=0.005, tau=10,
        )
        model_no = config_no.create_model()
        trainer_no = SparseTeacherForcingTrainer(model_no, config_no)
        trainer_no.train(data_padded, epochs=100)

        z_test = torch.randn(200, M)
        p_no = measure_invertibility(model_no, z_test)
        prop_no_reg.append(p_no)
        ds_no = compute_d_stsp(model_no, data_padded, n_steps=30)
        d_stsp_no.append(ds_no)
        print(f"  No reg: {p_no:.2%} inv, D_stsp={ds_no:.3f}")

        # With regularization
        torch.manual_seed(42)
        lambda_inv = 0.1 * np.exp(M / 20)
        config_reg = TrainingConfig(
            model_type="alrnn", M=M, H=None, P=P,
            sequence_length=50, noise_std=0.0,
            lambda_invert=lambda_inv,
            batch_size=4, epochs=100,
            learning_rate=0.005, tau=10,
        )
        model_reg = config_reg.create_model()
        trainer_reg = SparseTeacherForcingTrainer(
            model_reg, config_reg,
        )
        trainer_reg.train(data_padded, epochs=100)

        p_reg = measure_invertibility(model_reg, z_test)
        prop_with_reg.append(p_reg)
        ds_reg = compute_d_stsp(model_reg, data_padded, n_steps=30)
        d_stsp_reg.append(ds_reg)
        print(
            f"  With reg (λ={lambda_inv:.2f}):"
            f" {p_reg:.2%} inv, D_stsp={ds_reg:.3f}"
        )

    dims = np.array(dimensions)
    fig = plot_invertibility(
        dims,
        np.array(prop_no_reg),
        np.array(prop_with_reg),
        title="Fig 2: Invertibility Regularization",
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR, "fig2_invertibility.png"), dpi=150,
    )

    # Print D_stsp summary
    print("\n--- D_stsp Summary ---")
    print(f"{'M':>4s}  {'No Reg':>10s}  {'With Reg':>10s}")
    for i, M in enumerate(dimensions):
        print(f"{M:4d}  {d_stsp_no[i]:10.3f}  {d_stsp_reg[i]:10.3f}")

    # --- Damped oscillator convergence comparison ---
    print("\n--- Damped Oscillator Convergence ---")
    M_osc = 5
    P_osc = 3
    t_vals = np.linspace(0, 10, 200)
    osc_x = np.exp(-0.2 * t_vals) * np.cos(2 * t_vals)
    osc_y = np.exp(-0.2 * t_vals) * np.sin(2 * t_vals)
    osc_data = torch.zeros(200, M_osc)
    osc_data[:, 0] = torch.tensor(osc_x, dtype=torch.float32)
    osc_data[:, 1] = torch.tensor(osc_y, dtype=torch.float32)

    for label, lam in [("No reg", 0.0), ("With reg", 0.5)]:
        torch.manual_seed(42)
        cfg = TrainingConfig(
            model_type="alrnn", M=M_osc, H=None, P=P_osc,
            sequence_length=30, noise_std=0.0,
            lambda_invert=lam,
            batch_size=4, epochs=200,
            learning_rate=0.005, tau=10,
        )
        mdl = cfg.create_model()
        tr = SparseTeacherForcingTrainer(mdl, cfg)
        losses = tr.train(osc_data, epochs=200)
        ds = compute_d_stsp(mdl, osc_data, n_steps=20)
        print(
            f"  {label}: final loss={losses[-1]:.6f},"
            f" D_stsp={ds:.4f}"
        )

    print(f"\nAll saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_fig2()
