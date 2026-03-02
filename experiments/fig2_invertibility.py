"""Fig 2: Invertibility Regularization Ablation.

Trains ALRNN on Lorenz-63 with varying M and λ_invert.
Measures proportion of invertible subregions.
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


def run_fig2():
    """Run the invertibility ablation experiment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate Lorenz data
    print("Generating Lorenz-63 data...")
    traj = generate_trajectory(x0=np.array([1.0, 1.0, 1.0]), T=20.0, dt=0.01)
    data = torch.tensor(traj[:500], dtype=torch.float32)

    dimensions = [10, 20, 30]
    prop_no_reg = []
    prop_with_reg = []

    for M in dimensions:
        P = max(3, M // 3)
        print(f"\n--- M={M}, P={P} ---")

        # Without regularization
        torch.manual_seed(42)
        config_no = TrainingConfig(
            model_type="alrnn", M=M, H=None, P=P,
            sequence_length=50, noise_std=0.0, lambda_invert=0.0,
            batch_size=4, epochs=100, learning_rate=0.005, tau=10,
        )
        model_no = config_no.create_model()
        trainer_no = SparseTeacherForcingTrainer(model_no, config_no)
        # Pad data to match M dimensions
        data_padded = torch.zeros(data.shape[0], M)
        data_padded[:, :3] = data
        trainer_no.train(data_padded, epochs=100)

        z_test = torch.randn(200, M)
        p_no = measure_invertibility(model_no, z_test)
        prop_no_reg.append(p_no)
        print(f"  No reg: {p_no:.2%} invertible")

        # With regularization
        torch.manual_seed(42)
        lambda_inv = 0.1 * np.exp(M / 20)
        config_reg = TrainingConfig(
            model_type="alrnn", M=M, H=None, P=P,
            sequence_length=50, noise_std=0.0, lambda_invert=lambda_inv,
            batch_size=4, epochs=100, learning_rate=0.005, tau=10,
        )
        model_reg = config_reg.create_model()
        trainer_reg = SparseTeacherForcingTrainer(model_reg, config_reg)
        trainer_reg.train(data_padded, epochs=100)

        p_reg = measure_invertibility(model_reg, z_test)
        prop_with_reg.append(p_reg)
        print(f"  With reg (λ={lambda_inv:.2f}): {p_reg:.2%} invertible")

    dims = np.array(dimensions)
    fig = plot_invertibility(
        dims,
        np.array(prop_no_reg),
        np.array(prop_with_reg),
        title="Fig 2: Invertibility Regularization",
    )
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_invertibility.png"), dpi=150)
    print(f"\nSaved to {OUTPUT_DIR}/fig2_invertibility.png")


if __name__ == "__main__":
    run_fig2()
