"""2D Piecewise-Linear Map from Gardini et al. (2009).

F(X) = A_l · X + B  if x ≤ 0
F(X) = A_r · X + B  if x ≥ 0

Can be reformulated as a PLRNN (Eq 2) by defining:
    A = A_l,  W = [[τ_r - τ_l, 0], [-δ_r + δ_l, 0]],  h = B

Reference: Appendix H.1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PLMap:
    """2D piecewise-linear map.

    Parameters
    ----------
    tau_r, delta_r, tau_l, delta_l, c, d, h1, h2 : float
        Map parameters as defined in Gardini et al. (2009).
    """

    def __init__(
        self,
        tau_r: float,
        delta_r: float,
        tau_l: float,
        delta_l: float,
        c: float,
        d: float,
        h1: float,
        h2: float,
    ) -> None:
        self.tau_r = tau_r
        self.delta_r = delta_r
        self.tau_l = tau_l
        self.delta_l = delta_l
        self.c = c
        self.d = d
        self.h1 = h1
        self.h2 = h2

        self.A_l = np.array([[tau_l, c], [-delta_l, d]])
        self.A_r = np.array([[tau_r, c], [-delta_r, d]])
        self.B = np.array([h1, h2])

    def step(self, X: NDArray) -> NDArray:
        """Apply one step of the map.

        Parameters
        ----------
        X : ndarray
            State vector of shape ``(2,)``.

        Returns
        -------
        ndarray
            Next state of shape ``(2,)``.
        """
        A = self.A_r if X[0] > 0 else self.A_l
        return A @ X + self.B

    def trajectory(self, X0: NDArray, T: int) -> NDArray:
        """Generate a trajectory of length T.

        Parameters
        ----------
        X0 : ndarray
            Initial state of shape ``(2,)``.
        T : int
            Number of steps.

        Returns
        -------
        ndarray
            Trajectory of shape ``(T + 1, 2)``.
        """
        states = np.zeros((T + 1, 2))
        states[0] = X0
        for t in range(T):
            states[t + 1] = self.step(states[t])
        return states

    def to_plrnn_params(self) -> dict[str, NDArray]:
        """Convert to PLRNN parameters (A, W, h).

        The PLRNN form is: z_t = (A + W D(z)) z + h
        where D = diag(z > 0).

        Returns
        -------
        dict
            Keys ``'A'``, ``'W'``, ``'h'`` with numpy arrays.
        """
        A = self.A_l.copy()
        W = np.array([
            [self.tau_r - self.tau_l, 0.0],
            [-self.delta_r + self.delta_l, 0.0],
        ])
        h = self.B.copy()
        return {"A": A, "W": W, "h": h}

    def analytical_fixed_points(self) -> list[NDArray]:
        """Compute fixed points analytically.

        Fixed points satisfy z* = A_s z* + B where s ∈ {l, r}.
        So z* = (I - A_s)^{-1} B, checked for self-consistency.

        Returns
        -------
        list[ndarray]
            List of valid fixed points.
        """
        eye_2 = np.eye(2)
        fps = []

        for label, A in [("l", self.A_l), ("r", self.A_r)]:
            try:
                z_star = np.linalg.solve(eye_2 - A, self.B)
            except np.linalg.LinAlgError:
                continue

            # Self-consistency: check that z* is in the correct region
            if label == "l" and z_star[0] <= 0:
                fps.append(z_star)
            elif label == "r" and z_star[0] >= 0:
                fps.append(z_star)

        return fps

    # ------------------------------------------------------------------
    # Preset configurations from the paper
    # ------------------------------------------------------------------
    @classmethod
    def fig3a_left(cls) -> PLMap:
        """Parameters for Figure 3A (left)."""
        return cls(
            tau_r=1.2, delta_r=1.8, tau_l=-0.3, delta_l=0.9,
            c=-1.5, d=1.0, h2=-0.1, h1=-0.13,
        )

    @classmethod
    def fig3a_right(cls) -> PLMap:
        """Parameters for Figure 3A (right)."""
        return cls(
            tau_r=1.26, delta_r=0.71, tau_l=-0.39, delta_l=-0.91,
            c=-0.44, d=0.56, h2=0.62, h1=-0.28,
        )

    @classmethod
    def fig3b_left(cls) -> PLMap:
        """Parameters for Figure 3B (left) — period-4 cycle."""
        return cls(
            tau_r=-1.85, delta_r=0.9, tau_l=-0.3, delta_l=0.9,
            c=1.0, d=0.0, h2=0.0, h1=-1.0,
        )

    @classmethod
    def fig3b_right(cls) -> PLMap:
        """Parameters for Figure 3B (right) — period-3 cycle."""
        return cls(
            tau_r=-1.85, delta_r=0.9, tau_l=0.9, delta_l=0.9,
            c=1.0, d=0.0, h2=0.0, h1=-1.0,
        )

    @classmethod
    def fig5(cls) -> PLMap:
        """Parameters for Figure 5 — homoclinic chaos."""
        return cls(
            tau_r=1.5, delta_r=0.75, tau_l=-1.77, delta_l=0.9,
            c=0.6, d=0.15, h2=-0.4, h1=-0.7,
        )
