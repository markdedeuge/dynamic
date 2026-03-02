"""Lorenz-63 system.

    dx₁/dt = σ (x₂ - x₁)
    dx₂/dt = x₁ (ρ - x₃) - x₂
    dx₃/dt = x₁ x₂ - β x₃

Default parameters: σ=10, ρ=28, β=8/3.

Reference: Appendix H.3, Lorenz (1963).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# Default parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def ode_rhs(
    t: float,
    state: NDArray,
    sigma: float = SIGMA,
    rho: float = RHO,
    beta: float = BETA,
) -> NDArray:
    """Right-hand side of the Lorenz-63 ODE.

    Parameters
    ----------
    t : float
        Current time (unused, autonomous system).
    state : ndarray
        State vector ``[x1, x2, x3]`` of shape ``(3,)``.

    Returns
    -------
    ndarray
        Time derivative of shape ``(3,)``.
    """
    x1, x2, x3 = state
    dx1 = sigma * (x2 - x1)
    dx2 = x1 * (rho - x3) - x2
    dx3 = x1 * x2 - beta * x3
    return np.array([dx1, dx2, dx3])


def generate_trajectory(
    x0: NDArray,
    T: float,
    dt: float = 0.01,
    **kwargs: float,
) -> NDArray:
    """Integrate the Lorenz-63 system.

    Parameters
    ----------
    x0 : ndarray
        Initial state of shape ``(3,)``.
    T : float
        Total integration time.
    dt : float
        Time step for dense output.

    Returns
    -------
    ndarray
        Trajectory of shape ``(N, 3)``.
    """
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(
        ode_rhs,
        (0, T),
        x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        args=(
            kwargs.get("sigma", SIGMA),
            kwargs.get("rho", RHO),
            kwargs.get("beta", BETA),
        ),
    )
    return sol.y.T


def fixed_points(
    sigma: float = SIGMA,
    rho: float = RHO,
    beta: float = BETA,
) -> list[NDArray]:
    """Analytical fixed points of the Lorenz-63 system.

    Returns
    -------
    list[ndarray]
        Three fixed points: origin and the symmetric pair.
    """
    origin = np.array([0.0, 0.0, 0.0])
    c = np.sqrt(beta * (rho - 1))
    fp_plus = np.array([c, c, rho - 1])
    fp_minus = np.array([-c, -c, rho - 1])
    return [origin, fp_plus, fp_minus]
