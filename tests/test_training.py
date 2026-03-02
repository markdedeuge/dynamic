"""TDD tests for training infrastructure (Phase 3).

Tests written FIRST per TDD methodology. Modules:
- dynamic.training.losses (MSE, invertibility regularization)
- dynamic.training.trainer (sparse teacher forcing)
- dynamic.training.configs (Table 1 parameter configs)
"""

import numpy as np
import pytest
import torch

from dynamic.models.plrnn import PLRNN
from dynamic.models.shallow_plrnn import ShallowPLRNN


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
class TestLosses:
    """Tests for MSE loss and invertibility regularization."""

    def test_mse_loss_perfect(self):
        """Perfect predictions give loss = 0."""
        from dynamic.training.losses import mse_loss

        x = torch.randn(10, 3)
        loss = mse_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_mse_loss_value(self):
        """Known inputs give known MSE value."""
        from dynamic.training.losses import mse_loss

        pred = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0]])
        # MSE = (1/2) * ((1-0)^2 + (0-1)^2) / 1 = 1.0
        loss = mse_loss(pred, target)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_mse_loss_gradient_flows(self):
        """MSE loss has gradients w.r.t. predictions."""
        from dynamic.training.losses import mse_loss

        pred = torch.randn(5, 3, requires_grad=True)
        target = torch.randn(5, 3)
        loss = mse_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_invert_reg_negative_det(self):
        """Jacobian with det < 0 gives regularization > 0."""
        from dynamic.training.losses import invertibility_regularization

        torch.manual_seed(42)
        model = PLRNN(M=2)
        with torch.no_grad():
            model.A.copy_(torch.tensor([0.5, 0.5]))
            model.W.copy_(torch.tensor([[0.0, 2.0], [2.0, 0.0]]))

        z_test = torch.tensor([1.0, 1.0])
        J = model.get_jacobian(z_test)
        det = torch.det(J)
        assert det.item() < 0, f"Expected negative det, got {det.item()}"

        reg = invertibility_regularization(model, [J], lambda_inv=1.0)
        assert reg.item() > 0

    def test_invert_reg_positive_det(self):
        """All positive det Jacobians give regularization = 0."""
        from dynamic.training.losses import invertibility_regularization

        torch.manual_seed(42)
        model = PLRNN(M=2)
        with torch.no_grad():
            model.A.copy_(torch.tensor([1.0, 1.0]))
            model.W.copy_(torch.tensor([[0.0, 0.01], [0.01, 0.0]]))

        z_test = torch.tensor([1.0, 1.0])
        J = model.get_jacobian(z_test)
        reg = invertibility_regularization(model, [J], lambda_inv=1.0)
        assert reg.item() == pytest.approx(0.0, abs=1e-6)

    def test_invert_reg_gradient_flows(self):
        """Regularization has gradients w.r.t. model parameters."""
        from dynamic.training.losses import invertibility_regularization

        torch.manual_seed(42)
        model = PLRNN(M=3)
        z = torch.tensor([1.0, -1.0, 0.5])
        J = model.get_jacobian(z)
        reg = invertibility_regularization(model, [J], lambda_inv=1.0)
        reg.backward()
        assert model.A.grad is not None or model.W.grad is not None

    def test_invert_reg_multiple_regions(self):
        """Regularization averages over multiple subregions."""
        from dynamic.training.losses import invertibility_regularization

        torch.manual_seed(42)
        model = PLRNN(M=2)
        zs = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, -1.0]),
              torch.tensor([-1.0, 1.0])]
        jacobians = [model.get_jacobian(z) for z in zs]
        reg = invertibility_regularization(model, jacobians, lambda_inv=1.0)
        assert reg.item() >= 0

    def test_invert_reg_shplrnn(self):
        """Regularization works with ShallowPLRNN (uses get_jacobian)."""
        from dynamic.training.losses import invertibility_regularization

        torch.manual_seed(42)
        model = ShallowPLRNN(M=2, H=3)
        # Craft params so J = diag(A) + W1 @ D_h @ W2 has det < 0
        with torch.no_grad():
            model.A.copy_(torch.tensor([0.1, 0.1]))
            # Cross-coupling W1: hidden[0]→z[1], hidden[1]→z[0]
            model.W1.copy_(torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
            model.W2.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]))
            model.h2.zero_()

        z = torch.tensor([1.0, 1.0])  # all hidden positive → D_h = I
        J = model.get_jacobian(z)
        det = torch.det(J)
        assert det.item() < 0, f"Expected negative det, got {det.item()}"

        reg = invertibility_regularization(model, [J], lambda_inv=1.0)
        assert reg.item() > 0
        # Verify gradient flows through shPLRNN params
        reg.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "Gradient should flow through shPLRNN parameters"


# ---------------------------------------------------------------------------
# Training configs
# ---------------------------------------------------------------------------
class TestConfigs:
    """Tests for Table 1 parameter configurations."""

    def test_all_configs_exist(self):
        """All experiment configs exist and load."""
        from dynamic.training.configs import (
            DECISION_CONFIG,
            DUFFING_CONFIG,
            LORENZ_FIG2A_CONFIG,
            LORENZ_FIG4C_CONFIG,
            OSCILLATOR_CONFIG,
        )

        for config in [
            DUFFING_CONFIG,
            DECISION_CONFIG,
            LORENZ_FIG2A_CONFIG,
            LORENZ_FIG4C_CONFIG,
            OSCILLATOR_CONFIG,
        ]:
            assert config is not None

    def test_config_fields(self):
        """Each config has all required fields."""
        from dynamic.training.configs import DUFFING_CONFIG

        assert hasattr(DUFFING_CONFIG, "model_type")
        assert hasattr(DUFFING_CONFIG, "M")
        assert hasattr(DUFFING_CONFIG, "sequence_length")
        assert hasattr(DUFFING_CONFIG, "lambda_invert")
        assert hasattr(DUFFING_CONFIG, "batch_size")
        assert hasattr(DUFFING_CONFIG, "epochs")
        assert hasattr(DUFFING_CONFIG, "learning_rate")
        assert hasattr(DUFFING_CONFIG, "tau")

    def test_duffing_config_values(self):
        """Duffing config matches Table 1."""
        from dynamic.training.configs import DUFFING_CONFIG

        assert DUFFING_CONFIG.model_type == "shplrnn"
        assert DUFFING_CONFIG.M == 2
        assert DUFFING_CONFIG.H == 10
        assert DUFFING_CONFIG.batch_size == 32
        assert DUFFING_CONFIG.epochs == 10000
        assert DUFFING_CONFIG.learning_rate == 0.001
        assert DUFFING_CONFIG.tau == 15

    def test_config_create_model(self):
        """Config can create its corresponding model."""
        from dynamic.training.configs import DUFFING_CONFIG

        model = DUFFING_CONFIG.create_model()
        assert isinstance(model, ShallowPLRNN)
        assert model.M == 2
        assert model.H == 10


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class TestTrainer:
    """Tests for the sparse teacher forcing trainer."""

    def test_teacher_forcing_interval(self):
        """States are replaced every τ steps."""
        from dynamic.training.configs import TrainingConfig
        from dynamic.training.trainer import SparseTeacherForcingTrainer

        config = TrainingConfig(
            model_type="plrnn", M=3, H=None, P=None,
            sequence_length=20, noise_std=0.0, lambda_invert=0.0,
            batch_size=1, epochs=1, learning_rate=0.001, tau=5,
        )
        model = PLRNN(M=3)
        trainer = SparseTeacherForcingTrainer(model, config)

        # Generate a dummy sequence
        data = torch.randn(20, 3)
        forced_indices = trainer._get_forcing_indices(len(data))
        # Should force at steps 0, 5, 10, 15
        expected = [0, 5, 10, 15]
        assert forced_indices == expected

    def test_teacher_forcing_no_replace(self):
        """States between forcing indices evolve via model forward."""
        from dynamic.training.configs import TrainingConfig
        from dynamic.training.trainer import SparseTeacherForcingTrainer

        torch.manual_seed(42)
        config = TrainingConfig(
            model_type="plrnn", M=2, H=None, P=None,
            sequence_length=10, noise_std=0.0, lambda_invert=0.0,
            batch_size=1, epochs=1, learning_rate=0.001, tau=5,
        )
        model = PLRNN(M=2)
        trainer = SparseTeacherForcingTrainer(model, config)

        data = torch.randn(10, 2)
        predictions, _ = trainer._forward_with_forcing(data, data[0])

        # At forced index 0, z is set to data[0], then forward → predictions[0]
        expected_p0 = model.forward(data[0].detach())
        assert torch.allclose(predictions[0], expected_p0, atol=1e-6)

        # At non-forced index 1, z = predictions[0], then forward → predictions[1]
        expected_p1 = model.forward(predictions[0].detach())
        assert torch.allclose(predictions[1], expected_p1, atol=1e-6)

    def test_training_loss_decreases(self):
        """Loss decreases over training on simple data."""
        from dynamic.training.configs import TrainingConfig
        from dynamic.training.trainer import SparseTeacherForcingTrainer

        torch.manual_seed(42)
        config = TrainingConfig(
            model_type="plrnn", M=2, H=None, P=None,
            sequence_length=20, noise_std=0.0, lambda_invert=0.0,
            batch_size=4, epochs=50, learning_rate=0.005, tau=5,
        )
        model = PLRNN(M=2)
        trainer = SparseTeacherForcingTrainer(model, config)

        # Simple training data: sequences from a known system
        from dynamic.systems.duffing import generate_trajectory
        traj = generate_trajectory(
            x0=np.array([1.0, 0.0]), T=5.0, dt=0.01,
        )
        data = torch.tensor(traj[:100], dtype=torch.float32)

        losses = trainer.train(data, epochs=50)
        assert len(losses) == 50
        assert losses[-1] < losses[0], (
            f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_trainer_with_regularization(self):
        """Trainer works with invertibility regularization enabled."""
        from dynamic.training.configs import TrainingConfig
        from dynamic.training.trainer import SparseTeacherForcingTrainer

        torch.manual_seed(42)
        config = TrainingConfig(
            model_type="plrnn", M=2, H=None, P=None,
            sequence_length=20, noise_std=0.0, lambda_invert=0.5,
            batch_size=4, epochs=10, learning_rate=0.005, tau=5,
        )
        model = PLRNN(M=2)
        trainer = SparseTeacherForcingTrainer(model, config)

        data = torch.randn(50, 2)
        losses = trainer.train(data, epochs=10)
        assert len(losses) == 10
        assert all(np.isfinite(val) for val in losses)
