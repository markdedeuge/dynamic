"""Training parameter configurations from Table 1.

Each dataclass mirrors one column of Table 1 in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class TrainingConfig:
    """Configuration for PLRNN training.

    Attributes
    ----------
    model_type : str
        One of ``"plrnn"``, ``"shplrnn"``, ``"alrnn"``.
    M : int
        State-space dimensionality.
    H : int or None
        Hidden dimension (shPLRNN only).
    P : int or None
        Number of ReLU dimensions (ALRNN only).
    sequence_length : int
        Training sequence length.
    noise_std : float
        Gaussian noise standard deviation added to data.
    lambda_invert : float
        Invertibility regularization strength.
    batch_size : int
        Training batch size.
    epochs : int
        Number of training epochs.
    learning_rate : float
        Optimizer learning rate.
    tau : int
        Teacher forcing interval (replace every τ steps).
    """

    model_type: str
    M: int
    H: int | None
    P: int | None
    sequence_length: int
    noise_std: float
    lambda_invert: float
    batch_size: int
    epochs: int
    learning_rate: float
    tau: int

    def create_model(self) -> nn.Module:
        """Instantiate the model specified by this config.

        Returns
        -------
        nn.Module
            The PLRNN model.
        """
        if self.model_type == "plrnn":
            from dynamic.models.plrnn import PLRNN

            return PLRNN(M=self.M)
        elif self.model_type == "shplrnn":
            from dynamic.models.shallow_plrnn import ShallowPLRNN

            if self.H is None:
                msg = "shPLRNN requires H (hidden dimension)"
                raise ValueError(msg)
            return ShallowPLRNN(M=self.M, H=self.H)
        elif self.model_type == "alrnn":
            from dynamic.models.alrnn import ALRNN

            if self.P is None:
                msg = "ALRNN requires P (number of ReLU dimensions)"
                raise ValueError(msg)
            return ALRNN(M=self.M, P=self.P)
        else:
            msg = f"Unknown model type: {self.model_type}"
            raise ValueError(msg)


# ── Table 1 configs ──────────────────────────────────────────────────────────

DUFFING_CONFIG = TrainingConfig(
    model_type="shplrnn", M=2, H=10, P=None,
    sequence_length=100, noise_std=0.0, lambda_invert=0.0,
    batch_size=32, epochs=10000, learning_rate=0.001, tau=15,
)

DECISION_CONFIG = TrainingConfig(
    model_type="alrnn", M=15, H=None, P=6,
    sequence_length=100, noise_std=0.01, lambda_invert=0.2,
    batch_size=16, epochs=20000, learning_rate=0.005, tau=15,
)

LORENZ_FIG2A_CONFIG = TrainingConfig(
    model_type="alrnn", M=30, H=None, P=8,
    sequence_length=100, noise_std=0.0, lambda_invert=0.0,
    batch_size=16, epochs=1000, learning_rate=0.005, tau=15,
)

LORENZ_FIG4C_CONFIG = TrainingConfig(
    model_type="shplrnn", M=3, H=20, P=None,
    sequence_length=100, noise_std=0.05, lambda_invert=0.0,
    batch_size=16, epochs=1000, learning_rate=0.005, tau=15,
)

OSCILLATOR_CONFIG = TrainingConfig(
    model_type="alrnn", M=40, H=None, P=15,
    sequence_length=25, noise_std=0.0, lambda_invert=0.0,
    batch_size=16, epochs=1000, learning_rate=0.001, tau=10,
)
