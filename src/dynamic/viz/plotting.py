"""Plotting utilities for manifold visualization.

Provides publication-quality plots for state spaces, manifolds,
basins of attraction, bifurcation diagrams, and Lyapunov spectra.

All functions return matplotlib Figure objects for composability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy import ndarray
    from torch import Tensor

    from dynamic.analysis.manifolds import ManifoldSegment
    from dynamic.analysis.scyfi import FixedPoint


# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

# Color palette
STABLE_COLOR = "#2196F3"
UNSTABLE_COLOR = "#F44336"
SADDLE_COLOR = "#FF9800"
MANIFOLD_STABLE = "#4CAF50"
MANIFOLD_UNSTABLE = "#9C27B0"
TRAJECTORY_COLOR = "#607D8B"
BASIN_CMAP = "RdBu"


def plot_state_space_2d(
    trajectories: list[ndarray] | None = None,
    fixed_points: list[FixedPoint] | None = None,
    manifolds: list[ManifoldSegment] | None = None,
    manifold_type: str = "unstable",
    title: str = "State Space",
    ax: plt.Axes | None = None,
) -> Figure:
    """2D state space with trajectories, fixed points, and manifolds.

    Parameters
    ----------
    trajectories : list[ndarray]
        Trajectories to plot, each of shape ``(T, 2)``.
    fixed_points : list[FixedPoint]
        Fixed points with classification.
    manifolds : list[ManifoldSegment]
        Manifold segments to plot.
    manifold_type : str
        ``"stable"`` or ``"unstable"`` for color selection.
    title : str
        Plot title.
    ax : Axes or None
        Existing axes to plot on.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()

    # Trajectories
    if trajectories:
        for traj in trajectories:
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=TRAJECTORY_COLOR, alpha=0.3, linewidth=0.5,
            )

    # Manifold segments
    if manifolds:
        color = MANIFOLD_STABLE if manifold_type == "stable" else MANIFOLD_UNSTABLE
        for seg in manifolds:
            pts = seg.points.numpy() if hasattr(seg.points, "numpy") else seg.points
            if pts.ndim == 2 and pts.shape[0] > 0:
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    c=color, s=3, alpha=0.6, zorder=3,
                )

    # Fixed points
    if fixed_points:
        for fp in fixed_points:
            z = fp.z.numpy() if hasattr(fp.z, "numpy") else fp.z
            color_map = {
                "stable": STABLE_COLOR,
                "unstable": UNSTABLE_COLOR,
                "saddle": SADDLE_COLOR,
            }
            c = color_map.get(fp.classification, "black")
            marker = "o" if fp.classification == "stable" else (
                "^" if fp.classification == "unstable" else "s"
            )
            ax.plot(z[0], z[1], marker, color=c, markersize=10, zorder=5)

    ax.set_title(title)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    return fig


def plot_state_space_3d(
    trajectories: list[ndarray] | None = None,
    fixed_points: list[FixedPoint] | None = None,
    manifolds: list[ManifoldSegment] | None = None,
    dims: tuple[int, int, int] = (0, 1, 2),
    title: str = "State Space (3D)",
) -> Figure:
    """3D state space projection.

    Parameters
    ----------
    trajectories : list[ndarray]
        Trajectories of shape ``(T, M)``.
    fixed_points : list[FixedPoint]
        Fixed points.
    manifolds : list[ManifoldSegment]
        Manifold segments.
    dims : tuple
        3 dimensions to project onto.
    title : str
        Plot title.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    d0, d1, d2 = dims

    if trajectories:
        for traj in trajectories:
            ax.plot(
                traj[:, d0], traj[:, d1], traj[:, d2],
                color=TRAJECTORY_COLOR, alpha=0.3, linewidth=0.5,
            )

    if manifolds:
        for seg in manifolds:
            pts = seg.points.numpy() if hasattr(seg.points, "numpy") else seg.points
            if pts.ndim == 2 and pts.shape[1] > d2:
                ax.scatter(
                    pts[:, d0], pts[:, d1], pts[:, d2],
                    c=MANIFOLD_UNSTABLE, s=3, alpha=0.4,
                )

    if fixed_points:
        for fp in fixed_points:
            z = fp.z.numpy() if hasattr(fp.z, "numpy") else fp.z
            if len(z) > d2:
                ax.scatter(
                    [z[d0]], [z[d1]], [z[d2]],
                    c=SADDLE_COLOR, s=80, marker="s", zorder=5,
                )

    ax.set_title(title)
    ax.set_xlabel(f"$z_{{{d0 + 1}}}$")
    ax.set_ylabel(f"$z_{{{d1 + 1}}}$")
    ax.set_zlabel(f"$z_{{{d2 + 1}}}$")
    return fig


def plot_basins_2d(
    model,
    fixed_points: list,
    x_range: tuple[float, float] = (-3, 3),
    y_range: tuple[float, float] = (-3, 3),
    resolution: int = 100,
    T: int = 200,
    title: str = "Basins of Attraction",
) -> Figure:
    """Color-coded basins of attraction in 2D.

    Parameters
    ----------
    model : nn.Module
        PLRNN model.
    fixed_points : list
        Fixed points to classify basins towards.
    x_range : tuple
        X-axis limits.
    y_range : tuple
        Y-axis limits.
    resolution : int
        Grid resolution.
    T : int
        Iterations to determine convergence.
    title : str
        Plot title.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    import torch

    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    basin_map = np.zeros((resolution, resolution))

    stable_fps = [fp for fp in fixed_points if fp.classification == "stable"]

    with torch.no_grad():
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                z = torch.tensor([x, y], dtype=torch.float32)
                for _ in range(T):
                    z = model.forward(z)
                    if not torch.all(torch.isfinite(z)):
                        break

                if torch.all(torch.isfinite(z)):
                    # Assign to nearest stable FP
                    min_dist = float("inf")
                    for k, fp in enumerate(stable_fps):
                        fp_z = fp.z.detach()
                        dist = torch.norm(z - fp_z).item()
                        if dist < min_dist:
                            min_dist = dist
                            basin_map[j, i] = k + 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(
        basin_map, extent=[*x_range, *y_range],
        origin="lower", cmap=BASIN_CMAP, alpha=0.7,
    )

    for fp in fixed_points:
        z = fp.z.detach().numpy()
        c = {"stable": STABLE_COLOR, "unstable": UNSTABLE_COLOR,
             "saddle": SADDLE_COLOR}.get(fp.classification, "black")
        ax.plot(z[0], z[1], "o" if fp.classification == "stable" else "s",
                color=c, markersize=10, zorder=5)

    ax.set_title(title)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    return fig


def plot_manifold_quality(
    deltas_on: list[float],
    deltas_off: list[float],
    title: str = "Manifold Quality",
) -> Figure:
    """δ_σ histograms: on-manifold vs off-manifold.

    Parameters
    ----------
    deltas_on : list[float]
        δ_σ values for on-manifold points.
    deltas_off : list[float]
        δ_σ values for random off-manifold points.
    title : str
        Plot title.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(0, 1, 50)
    ax.hist(deltas_on, bins=bins, alpha=0.7, color=MANIFOLD_STABLE,
            label="On manifold", density=True)
    ax.hist(deltas_off, bins=bins, alpha=0.7, color=UNSTABLE_COLOR,
            label="Off manifold", density=True)
    ax.set_xlabel("$\\delta_\\sigma$")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_bifurcation(
    param_values: ndarray,
    attractor_values: list[ndarray],
    param_name: str = "$h_1$",
    title: str = "Bifurcation Diagram",
) -> Figure:
    """Bifurcation diagram: attractor values vs parameter.

    Parameters
    ----------
    param_values : ndarray
        Parameter values.
    attractor_values : list[ndarray]
        For each param, the attractor x-values visited.
    param_name : str
        Parameter label.
    title : str
        Plot title.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for p, vals in zip(param_values, attractor_values):
        ax.plot(np.full(len(vals), p), vals, ",", color="black", alpha=0.1)
    ax.set_xlabel(param_name)
    ax.set_ylabel("$x_1$")
    ax.set_title(title)
    return fig


def plot_lyapunov_spectrum(
    param_values: ndarray,
    exponents: ndarray,
    param_name: str = "$h_1$",
    title: str = "Lyapunov Spectrum",
) -> Figure:
    """Lyapunov exponents vs parameter.

    Parameters
    ----------
    param_values : ndarray
        Parameter values.
    exponents : ndarray
        Shape ``(N_params, M)`` Lyapunov exponents.
    param_name : str
        Parameter label.
    title : str
        Plot title.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(exponents.shape[1]):
        ax.plot(param_values, exponents[:, i], linewidth=1.5, label=f"$\\lambda_{i + 1}$")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_invertibility(
    dimensions: ndarray,
    prop_no_reg: ndarray,
    prop_with_reg: ndarray,
    title: str = "Invertibility vs Dimension",
) -> Figure:
    """Invertibility proportion vs model dimension.

    Parameters
    ----------
    dimensions : ndarray
        Model dimensions tested.
    prop_no_reg : ndarray
        Proportion invertible without regularization.
    prop_with_reg : ndarray
        Proportion invertible with regularization.
    title : str
        Plot title.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(dimensions, prop_no_reg, "o-", label="No regularization",
            color=UNSTABLE_COLOR, linewidth=2)
    ax.plot(dimensions, prop_with_reg, "s-", label="With regularization",
            color=MANIFOLD_STABLE, linewidth=2)
    ax.set_xlabel("Dimension M")
    ax.set_ylabel("Proportion det(J) > 0")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    return fig
