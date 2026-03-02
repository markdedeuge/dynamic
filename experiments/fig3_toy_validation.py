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
from dynamic.analysis.quality import delta_sigma_statistic
from dynamic.analysis.scyfi import FixedPoint
from dynamic.analysis.subregions import classify_point
from dynamic.systems.pl_map import PLMap
from dynamic.viz.plotting import plot_state_space_2d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class PLMapModel(torch.nn.Module):
    """Thin nn.Module wrapper around PL map (A + WD) z + h.

    Uses the full A matrix from to_plrnn_params(), unlike the diagonal-A
    PLRNN class. Provides the same get_jacobian/get_D/get_subregion_id API.
    """

    def __init__(self, pl_map: PLMap):
        super().__init__()
        params = pl_map.to_plrnn_params()
        self.A = torch.nn.Parameter(
            torch.as_tensor(params["A"], dtype=torch.float32),
        )
        self.W = torch.nn.Parameter(
            torch.as_tensor(params["W"], dtype=torch.float32),
        )
        self.h = torch.nn.Parameter(
            torch.as_tensor(params["h"], dtype=torch.float32),
        )
        self.M = self.h.shape[0]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        D = torch.diag((z > 0).float())
        return (self.A + self.W @ D) @ z + self.h

    def get_D(self, z: torch.Tensor) -> torch.Tensor:
        return torch.diag((z > 0).float())

    def get_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        D = self.get_D(z)
        return self.A + self.W @ D

    def get_subregion_id(self, z: torch.Tensor) -> tuple:
        return tuple(int(x > 0) for x in z.tolist())

    def forward_trajectory(self, z0: torch.Tensor, T: int) -> torch.Tensor:
        traj = [z0]
        z = z0
        with torch.no_grad():
            for _ in range(T):
                z = self.forward(z)
                traj.append(z)
        return torch.stack(traj)


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


if __name__ == "__main__":
    run_fig3()
