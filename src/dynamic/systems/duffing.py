"""Duffing oscillator.

    ẍ + δ ẋ + α x + β x³ = γ cos(ωt)

Default parameters (unforced bistable): α=-1, β=0.1, δ=0.5, γ=0.

Reference: Appendix H.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# Default parameters from the paper
ALPHA = -1.0
BETA = 0.1
DELTA = 0.5
GAMMA = 0.0
OMEGA = 1.0


def ode_rhs(
    t: float,
    state: NDArray,
    alpha: float = ALPHA,
    beta: float = BETA,
    delta: float = DELTA,
    gamma: float = GAMMA,
    omega: float = OMEGA,
) -> NDArray:
    """Right-hand side of the Duffing ODE.

    State = [x, v] where v = dx/dt.

    Parameters
    ----------
    t : float
        Current time.
    state : ndarray
        State vector ``[x, v]`` of shape ``(2,)``.

    Returns
    -------
    ndarray
        Time derivative ``[dx/dt, dv/dt]``.
    """
    x, v = state
    dx = v
    dv = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return np.array([dx, dv])


def generate_trajectory(
    x0: NDArray,
    T: float,
    dt: float = 0.01,
    **kwargs: float,
) -> NDArray:
    """Integrate the Duffing system.

    Parameters
    ----------
    x0 : ndarray
        Initial state ``[x, v]`` of shape ``(2,)``.
    T : float
        Total integration time.
    dt : float
        Time step for dense output.

    Returns
    -------
    ndarray
        Trajectory of shape ``(N, 2)``.
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
            kwargs.get("alpha", ALPHA),
            kwargs.get("beta", BETA),
            kwargs.get("delta", DELTA),
            kwargs.get("gamma", GAMMA),
            kwargs.get("omega", OMEGA),
        ),
    )
    return sol.y.T
