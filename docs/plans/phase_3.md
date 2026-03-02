# Phase 3: Training Infrastructure

**Status**: ✅ Complete
**Depends on**: Phase 1 (models), Phase 2 (systems for data)
**Estimated effort**: Medium

## Objective

Implement sparse teacher forcing training and invertibility regularization as described in §3.4 and Appendix E.

## Key Concepts

### Sparse Teacher Forcing (Mikhaeil et al., 2022)

Every τ time steps, replace the latent state `z_t` with a data-inferred state `ẑ_t = g⁻¹(x_t)`. For all experiments in this paper, `g` is the identity, so `ẑ_t = x_t`.

### Invertibility Regularization (Eq 5)

$$\mathcal{L}_{\text{reg}} = \lambda \cdot \frac{1}{|\mathcal{S}_{\text{reg}}|} \sum_{i \in \mathcal{S}_{\text{reg}}} \max(0, -\det(J_i))$$

Sample 1% of traversed subregions. Penalizes negative Jacobian determinants to enforce map invertibility.

### Training Configs (Table 1)

| Parameter | Duffing | Decision | Lorenz (Fig 4C) |
|-----------|---------|----------|-----------------|
| Model | shPLRNN | ALRNN | shPLRNN |
| M | 2 | 15 | 3 |
| H / P | H=10 | P=6 | H=20 |
| Seq length | 100 | 100 | 100 |
| λ_invert | 0.0 | 0.2 | 0.0 |
| Batch size | 32 | 16 | 16 |
| Epochs | 10000 | 20000 | 1000 |
| LR | 0.001 | 0.005 | 0.005 |
| τ | 15 | 15 | 15 |

## TDD Tests — `tests/test_training.py`

| Test | Assertion |
|------|-----------|
| `test_mse_loss_perfect` | Perfect predictions → loss exactly 0 |
| `test_mse_loss_value` | Known inputs → known MSE value |
| `test_invert_reg_negative_det` | Jacobian with det < 0 → reg > 0 |
| `test_invert_reg_positive_det` | All dets > 0 → reg = 0 |
| `test_invert_reg_gradient_flows` | Regularization term has gradients w.r.t. model params |
| `test_teacher_forcing_interval` | States are replaced at exactly every τ steps |
| `test_teacher_forcing_no_replace` | States between replacements evolve via model forward |
| `test_training_loss_decreases` | 100 epochs on Duffing → final loss < initial loss |
| `test_config_loads` | All 5 experiment configs (from Table 1) instantiate correctly |
| `test_config_fields` | Each config has: model, M, H/P, lr, epochs, τ, λ_invert |

## Files

### [NEW] `src/dynamic/training/losses.py`

```python
def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor
def invertibility_regularization(
    model: nn.Module,
    jacobians: list[Tensor],
    lambda_inv: float,
) -> Tensor
```

### [NEW] `src/dynamic/training/trainer.py`

```python
class SparseTeacherForcingTrainer:
    def __init__(self, model, config)
    def train(self, data: Tensor, epochs: int) -> list[float]  # loss history
    def _forward_with_forcing(self, data: Tensor, z0: Tensor) -> tuple[Tensor, Tensor]
    def _collect_jacobians(self, trajectory: Tensor) -> list[Tensor]
    def _get_forcing_indices(self, seq_len: int) -> list[int]
```

### [NEW] `src/dynamic/training/configs.py`

```python
@dataclass
class TrainingConfig:
    model_type: str  # "plrnn" | "shplrnn" | "alrnn"
    M: int
    H: int | None  # for shPLRNN
    P: int | None  # for ALRNN
    sequence_length: int
    noise_std: float
    lambda_invert: float
    batch_size: int
    epochs: int
    learning_rate: float
    tau: int  # teacher forcing interval

DUFFING_CONFIG = TrainingConfig(...)
DECISION_CONFIG = TrainingConfig(...)
# etc.
```

## Verification

```bash
python -m pytest tests/test_training.py -v
ruff check src/dynamic/training/
```
