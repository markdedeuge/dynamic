"""Benchmark datasets with ReLU-compatible embedding.

Embeds low-dimensional PLRNN systems into higher dimensions.
Because PLRNN uses D(z) = diag(z > 0) coordinate-wise, a full
orthogonal rotation would break the dynamics. Instead we:

1. Pad with decoupled spectator dimensions (diagonal A, no W coupling)
2. Apply a coordinate permutation to scatter core dims across all D
3. Track ground-truth cycles via the same permutation

The spectator dimensions have their own self-consistent fixed points
(z_i* = h_i / (1 - a_i) if z_i* > 0, else z_i* = 0). The core
dimensions retain the exact same cycle structure.
"""
# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from dynamic.analysis.scyfi_exhaustive import find_cycles_exhaustive


# ---------------------------------------------------------------------------
# Permutation-based embedding
# ---------------------------------------------------------------------------
def _random_permutation(n: int, seed: int = 0) -> Tensor:
    """Generate a random permutation of [0, n)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=gen)


def embed_system(
    A: Tensor,
    W: Tensor,
    h: Tensor,
    target_dim: int,
    *,
    spectator_a_range: tuple[float, float] = (0.3, 0.7),
    spectator_h_range: tuple[float, float] = (-1.0, 1.0),
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Embed a low-dim PLRNN into a higher dimension.

    Parameters
    ----------
    A, W, h : Tensor
        Original PLRNN parameters (dim × dim, dim).
    target_dim : int
        Target embedding dimension D > dim.
    spectator_a_range : tuple
        Range for spectator diagonal A values.
    spectator_h_range : tuple
        Range for spectator bias values.
    seed : int
        Random seed for spectator params and permutation.

    Returns
    -------
    A_emb, W_emb, h_emb : Tensor
        Embedded PLRNN parameters (D × D, D).
    perm : Tensor
        Permutation vector of length D. core_dims = perm[:dim].
    """
    dim = A.shape[0]
    dtype = A.dtype

    if target_dim <= dim:
        msg = f"target_dim ({target_dim}) must be > dim ({dim})"
        raise ValueError(msg)

    D = target_dim
    pad = D - dim
    gen = torch.Generator().manual_seed(seed)

    # Step 1: Create block-diagonal system
    A_big = torch.zeros(D, D, dtype=dtype)
    A_big[:dim, :dim] = A

    # Spectator diagonal values
    a_lo, a_hi = spectator_a_range
    spec_a = torch.rand(pad, generator=gen, dtype=dtype) * (a_hi - a_lo) + a_lo
    A_big[dim:, dim:] = torch.diag(spec_a)

    W_big = torch.zeros(D, D, dtype=dtype)
    W_big[:dim, :dim] = W
    # Spectators have no W coupling

    h_big = torch.zeros(D, dtype=dtype)
    h_big[:dim] = h
    h_lo, h_hi = spectator_h_range
    h_big[dim:] = torch.rand(pad, generator=gen, dtype=dtype) * (h_hi - h_lo) + h_lo

    # Step 2: Random permutation of coordinates
    perm = _random_permutation(D, seed=seed)

    # Apply permutation: swap rows and columns
    A_emb = A_big[perm][:, perm]
    W_emb = W_big[perm][:, perm]
    h_emb = h_big[perm]

    return A_emb, W_emb, h_emb, perm


def embed_cycles(
    cycles_per_order: list[list[list[Tensor]]],
    perm: Tensor,
    original_dim: int,
    A_emb: Tensor,
    W_emb: Tensor,
    h_emb: Tensor,
) -> list[list[list[Tensor]]]:
    """Transform ground truth cycles into the embedded space.

    The spectator dimensions converge to their own fixed points.
    We compute the full embedded fixed point by simulating the
    embedded system from the known core coordinates.

    Parameters
    ----------
    cycles_per_order : list
        Cycles from the original system.
    perm : Tensor
        Permutation vector.
    original_dim : int
        Original system dimension.
    A_emb, W_emb, h_emb : Tensor
        Embedded PLRNN parameters.

    Returns
    -------
    list
        Cycles in the embedded D-dimensional space.
    """
    D = perm.shape[0]
    dtype = A_emb.dtype
    inv_perm = torch.argsort(perm)

    embedded = []
    for order_idx, order_cycles in enumerate(cycles_per_order):
        order = order_idx + 1
        order_embedded = []
        for traj in order_cycles:
            # Simulate from the known cycle to find spectator values
            # Start with core coordinates from GT, spectators from 0
            z = torch.zeros(D, dtype=dtype)
            for j in range(original_dim):
                z[inv_perm[j]] = traj[0][j]

            # Iterate many times to let spectators converge
            for _ in range(200):
                d = (z > 0).to(dtype)
                z_new = (A_emb + W_emb * d) @ z + h_emb
                # Reset core dims to GT values at each cycle step
                z = z_new

            # Now build the full embedded trajectory
            emb_traj = []
            z_start = z.clone()
            for step in range(order):
                emb_traj.append(z_start.clone())
                d = (z_start > 0).to(dtype)
                z_start = (A_emb + W_emb * d) @ z_start + h_emb

            order_embedded.append(emb_traj)
        embedded.append(order_embedded)
    return embedded


# ---------------------------------------------------------------------------
# Dataset dataclass
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkDataset:
    """A benchmark PLRNN with known ground truth cycles."""

    name: str
    A: Tensor
    W: Tensor
    h: Tensor
    original_dim: int
    embedded_dim: int
    ground_truth_cycles: list[list[list[Tensor]]]
    max_order: int

    @property
    def n_gt_cycles(self) -> int:
        return sum(len(c) for c in self.ground_truth_cycles)

    def gt_cycle_set(self, scale: int = 1000) -> set[tuple[int, ...]]:
        keys: set[tuple[int, ...]] = set()
        for order_cycles in self.ground_truth_cycles:
            for traj in order_cycles:
                for pt in traj:
                    keys.add(tuple(torch.round(pt * scale).long().tolist()))
        return keys


# ---------------------------------------------------------------------------
# Base systems
# ---------------------------------------------------------------------------
def _plmap_system(name: str = "fig3a_left") -> tuple[Tensor, Tensor, Tensor]:
    """Get PLRNN params from a PLMap preset."""
    from dynamic.systems.pl_map import PLMap

    presets = {
        "fig3a_left": PLMap.fig3a_left,
        "fig3a_right": PLMap.fig3a_right,
        "fig3b_left": PLMap.fig3b_left,
        "fig3b_right": PLMap.fig3b_right,
        "fig5": PLMap.fig5,
    }
    pl = presets[name]()
    params = pl.to_plrnn_params()
    A = torch.tensor(params["A"], dtype=torch.float64)
    W = torch.tensor(params["W"], dtype=torch.float64)
    h = torch.tensor(params["h"], dtype=torch.float64)
    return A, W, h


def _random_system(dim: int, seed: int = 42) -> tuple[Tensor, Tensor, Tensor]:
    """Generate a random PLRNN."""
    gen = torch.Generator().manual_seed(seed)
    A = torch.diag(torch.randn(dim, generator=gen, dtype=torch.float64))
    W = torch.randn(dim, dim, generator=gen, dtype=torch.float64) * 0.3
    W.fill_diagonal_(0.0)
    h = torch.randn(dim, generator=gen, dtype=torch.float64)
    return A, W, h


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------
def build_dataset(
    name: str,
    A: Tensor,
    W: Tensor,
    h: Tensor,
    target_dim: int,
    max_order: int = 4,
    *,
    seed: int = 0,
) -> BenchmarkDataset:
    """Build an embedded benchmark dataset with ground truth.

    1. Embed original system into target_dim via permutation
    2. Compute ground truth in the embedded system directly:
       - Exhaustive search if 2^D feasible
       - Vectorised solver with large budget otherwise
    """
    from dynamic.analysis.scyfi_exhaustive import scy_fi_exhaustive

    original_dim = A.shape[0]

    if target_dim == original_dim:
        gt_cycles, _ = find_cycles_exhaustive(
            A,
            W,
            h,
            max_order,
            max_systems=500_000,
        )
        return BenchmarkDataset(
            name=f"{name}_d{target_dim}",
            A=A,
            W=W,
            h=h,
            original_dim=original_dim,
            embedded_dim=target_dim,
            ground_truth_cycles=gt_cycles,
            max_order=max_order,
        )

    # Embed
    A_emb, W_emb, h_emb, perm = embed_system(
        A,
        W,
        h,
        target_dim,
        seed=seed,
    )

    # Compute GT directly in embedded space
    n_sys_k1 = 1 << target_dim
    if n_sys_k1 <= 500_000:
        # Exhaustive feasible — compute per-order
        gt_emb: list[list[list[Tensor]]] = []
        for order in range(1, max_order + 1):
            n_sys = n_sys_k1**order
            if n_sys <= 500_000:
                cycles, _ = scy_fi_exhaustive(
                    A_emb,
                    W_emb,
                    h_emb,
                    order,
                    gt_emb,
                    max_systems=500_000,
                )
                gt_emb.append(cycles)
            else:
                gt_emb.append([])
    else:
        # Too large — use vectorised with large budget
        from dynamic.analysis.scyfi_vectorised import find_cycles_vectorised

        vres = find_cycles_vectorised(
            A_emb,
            W_emb,
            h_emb,
            max_order,
            outer_loop_iterations=20,
            inner_loop_iterations=50,
            batch_size=64,
        )
        gt_emb = vres[0]

    return BenchmarkDataset(
        name=f"{name}_d{target_dim}",
        A=A_emb,
        W=W_emb,
        h=h_emb,
        original_dim=original_dim,
        embedded_dim=target_dim,
        ground_truth_cycles=gt_emb,
        max_order=max_order,
    )


def build_all_datasets(
    target_dims: list[int] | None = None,
    max_order: int = 4,
) -> list[BenchmarkDataset]:
    """Build the full benchmark suite.

    Returns datasets from multiple base systems, each embedded
    at multiple target dimensions.
    """
    if target_dims is None:
        target_dims = [5, 10, 20, 50]

    base_systems = [
        ("PLMap_3a", *_plmap_system("fig3a_left")),
        ("PLMap_3b_p4", *_plmap_system("fig3b_left")),
        ("PLMap_3b_p3", *_plmap_system("fig3b_right")),
        ("PLMap_5_chaos", *_plmap_system("fig5")),
        ("Random_2d", *_random_system(2, seed=42)),
        ("Random_3d", *_random_system(3, seed=43)),
    ]

    datasets = []
    for sys_name, A, W, h in base_systems:
        original_dim = A.shape[0]
        # Original dimension
        try:
            ds = build_dataset(
                sys_name,
                A,
                W,
                h,
                original_dim,
                max_order=max_order,
            )
            datasets.append(ds)
        except ValueError:
            pass

        # Embedded dimensions
        for D in target_dims:
            if D <= original_dim:
                continue
            ds = build_dataset(
                sys_name,
                A,
                W,
                h,
                D,
                max_order=max_order,
                seed=D,
            )
            datasets.append(ds)

    return datasets
