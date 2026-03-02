"""Sparse teacher forcing trainer for PLRNNs.

Replaces latent states z_t with data-inferred states every τ steps,
following Mikhaeil et al. (2022). Uses the identity observation model
g(z) = z for all experiments in this paper.

Reference: Appendix E.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dynamic.training.configs import TrainingConfig
from dynamic.training.losses import invertibility_regularization, mse_loss


class SparseTeacherForcingTrainer:
    """Train a PLRNN with sparse teacher forcing.

    Parameters
    ----------
    model : nn.Module
        A PLRNN, ShallowPLRNN, or ALRNN model.
    config : TrainingConfig
        Training hyperparameters.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )

    def _get_forcing_indices(self, seq_len: int) -> list[int]:
        """Indices at which to apply teacher forcing.

        Parameters
        ----------
        seq_len : int
            Length of the training sequence.

        Returns
        -------
        list[int]
            Indices where state replacement occurs.
        """
        return list(range(0, seq_len, self.config.tau))

    def _collect_jacobians(
        self, trajectory: Tensor, sample_frac: float = 0.01
    ) -> list[Tensor]:
        """Sample Jacobian matrices from traversed subregions.

        For invertibility regularization, sample a small fraction
        of the subregions visited during a trajectory.

        Parameters
        ----------
        trajectory : Tensor
            Trajectory of shape ``(T, M)``.
        sample_frac : float
            Fraction of time steps to sample.

        Returns
        -------
        list[Tensor]
            List of Jacobian matrices.
        """
        n_samples = max(1, int(len(trajectory) * sample_frac))
        indices = torch.randperm(len(trajectory))[:n_samples]
        jacobians = []
        for idx in indices:
            z = trajectory[idx]
            jacobians.append(self.model.get_jacobian(z))
        return jacobians

    def _forward_with_forcing(
        self, data: Tensor, z0: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run forward pass with sparse teacher forcing.

        Parameters
        ----------
        data : Tensor
            Training data of shape ``(T, M)``.
        z0 : Tensor
            Initial latent state.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predictions and the full latent trajectory.
        """
        forcing_indices = set(self._get_forcing_indices(len(data)))
        predictions = []
        z = z0

        for t in range(len(data)):
            if t in forcing_indices and t < len(data):
                z = data[t].detach()
            z_next = self.model.forward(z)
            predictions.append(z_next)
            z = z_next

        return torch.stack(predictions), z

    def train(self, data: Tensor, epochs: int | None = None) -> list[float]:
        """Train the model on data.

        Parameters
        ----------
        data : Tensor
            Training data of shape ``(T, M)``.
        epochs : int or None
            Number of epochs (overrides config if provided).

        Returns
        -------
        list[float]
            Loss history per epoch.
        """
        n_epochs = epochs or self.config.epochs
        seq_len = self.config.sequence_length
        loss_history = []

        for _epoch in range(n_epochs):
            self.optimizer.zero_grad()

            # Sample a random subsequence for this epoch
            if len(data) > seq_len + 1:
                start = torch.randint(0, len(data) - seq_len, (1,)).item()
                chunk = data[start : start + seq_len]
            else:
                chunk = data

            # Forward with teacher forcing
            z0 = chunk[0].detach()
            predictions, _ = self._forward_with_forcing(chunk, z0)

            # Targets are the next states (offset by 1)
            targets = chunk[1:]
            preds = predictions[: len(targets)]

            # MSE loss
            loss = mse_loss(preds, targets)

            # Invertibility regularization
            if self.config.lambda_invert > 0:
                jacobians = self._collect_jacobians(predictions.detach())
                reg = invertibility_regularization(
                    self.model, jacobians, self.config.lambda_invert
                )
                loss = loss + reg

            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())

        return loss_history
