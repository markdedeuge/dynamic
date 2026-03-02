"""10D damped nonlinear oscillator.

    ẋᵢ = yᵢ
    ẏᵢ = -αᵢ xᵢ - βᵢ xᵢ³ - γᵢ yᵢ     i = 1, ..., M

State vector: [x₁, ..., x_M, y₁, ..., y_M] of dimension 2M.

Reference: Appendix H.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# Parameters from the paper (M=10)
ALPHA = np.array([1.00, 0.80, 0.90, 0.30, 0.30, 0.25, 0.50, 0.89, 0.73, 0.21])
BETA_OSC = np.array([
    -0.02, -0.10, 0.01, -0.01, -0.01, 0.01, -0.06, -0.052, 0.03, -0.012,
])
GAMMA = np.array([0.10, 0.05, 0.12, 0.07, 0.00, 0.03, 0.09, 0.04, 0.06, 0.008])

M_DEFAULT = 10


def ode_rhs(
    t: float,
    state: NDArray,
    alpha: NDArray = ALPHA,
    beta: NDArray = BETA_OSC,
    gamma: NDArray = GAMMA,
) -> NDArray:
    """Right-hand side of the oscillator system.

    Parameters
    ----------
    t : float
        Current time (unused).
    state : ndarray
        State ``[x₁..x_M, y₁..y_M]`` of shape ``(2M,)``.

    Returns
    -------
    ndarray
        Time derivative of shape ``(2M,)``.
    """
    M = len(alpha)
    x = state[:M]
    y = state[M:]
    dx = y.copy()
    dy = -alpha * x - beta * x**3 - gamma * y
    return np.concatenate([dx, dy])


def generate_trajectory(
    x0: NDArray,
    T: float,
    dt: float = 0.01,
    **kwargs: NDArray,
) -> NDArray:
    """Integrate the oscillator system.

    Parameters
    ----------
    x0 : ndarray
        Initial state of shape ``(2M,)``.
    T : float
        Total integration time.
    dt : float
        Time step for dense output.

    Returns
    -------
    ndarray
        Trajectory of shape ``(N, 2M)``.
    """
    alpha = kwargs.get("alpha", ALPHA)
    beta = kwargs.get("beta", BETA_OSC)
    gamma = kwargs.get("gamma", GAMMA)

    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(
        ode_rhs,
        (0, T),
        x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        args=(alpha, beta, gamma),
    )
    return sol.y.T
