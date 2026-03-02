"""Multistable decision-making model from Gerstner et al. (2014).

    τ_E dh_E1/dt = -h_E1 + w_EE g_E(h_E1) + w_EI γ h_inh + R I₁
    τ_E dh_E2/dt = -h_E2 + w_EE g_E(h_E2) + w_EI γ h_inh + R I₂
    τ_inh dh_inh/dt = -h_inh + w_IE (g_E(h_E1) + g_E(h_E2))

where g_E(h; θ) = 1 / (1 + exp(-(h - θ))).

Reference: Appendix H.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# Default parameters from the paper
TAU_E = 10.0
TAU_INH = 5.0
W_EE = 16.0
W_EI = -15.0
W_IE = 12.0
R = 1.0
I1 = 40.0
I2 = 40.0
GAMMA_DEC = 1.0
THETA = 5.0


def sigmoid(h: float | NDArray, theta: float = THETA) -> float | NDArray:
    """Sigmoid activation function g_E(h; θ).

    Parameters
    ----------
    h : float or ndarray
        Input value(s).
    theta : float
        Threshold parameter.

    Returns
    -------
    float or ndarray
        Sigmoid output.
    """
    return 1.0 / (1.0 + np.exp(-(h - theta)))


def ode_rhs(
    t: float,
    state: NDArray,
    tau_e: float = TAU_E,
    tau_inh: float = TAU_INH,
    w_ee: float = W_EE,
    w_ei: float = W_EI,
    w_ie: float = W_IE,
    r: float = R,
    i1: float = I1,
    i2: float = I2,
    gamma: float = GAMMA_DEC,
    theta: float = THETA,
) -> NDArray:
    """Right-hand side of the decision-making ODE.

    Parameters
    ----------
    t : float
        Current time (unused).
    state : ndarray
        State ``[h_E1, h_E2, h_inh]`` of shape ``(3,)``.

    Returns
    -------
    ndarray
        Time derivative of shape ``(3,)``.
    """
    h_e1, h_e2, h_inh = state
    g1 = sigmoid(h_e1, theta)
    g2 = sigmoid(h_e2, theta)

    dh_e1 = (-h_e1 + w_ee * g1 + w_ei * gamma * h_inh + r * i1) / tau_e
    dh_e2 = (-h_e2 + w_ee * g2 + w_ei * gamma * h_inh + r * i2) / tau_e
    dh_inh = (-h_inh + w_ie * (g1 + g2)) / tau_inh

    return np.array([dh_e1, dh_e2, dh_inh])


def generate_trajectory(
    x0: NDArray,
    T: float,
    dt: float = 0.01,
    **kwargs: float,
) -> NDArray:
    """Integrate the decision-making system.

    Parameters
    ----------
    x0 : ndarray
        Initial state ``[h_E1, h_E2, h_inh]`` of shape ``(3,)``.
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
            kwargs.get("tau_e", TAU_E),
            kwargs.get("tau_inh", TAU_INH),
            kwargs.get("w_ee", W_EE),
            kwargs.get("w_ei", W_EI),
            kwargs.get("w_ie", W_IE),
            kwargs.get("r", R),
            kwargs.get("i1", I1),
            kwargs.get("i2", I2),
            kwargs.get("gamma", GAMMA_DEC),
            kwargs.get("theta", THETA),
        ),
    )
    return sol.y.T
