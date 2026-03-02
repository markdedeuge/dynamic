# Phase 1: PLRNN Models

**Status**: ✅ Complete
**Depends on**: Phase 0
**Estimated effort**: Medium

## Objective

Implement the three PLRNN architectures from §3.1 of the paper as PyTorch `nn.Module` classes, along with subregion utilities for the piecewise-linear structure.

## Reference Equations

**Standard PLRNN** (Eq 1):
$$z_t = A z_{t-1} + W \Phi(z_{t-1}) + h$$

**PL form** (Eq 2):
$$z_t = (A + W D_{t-1}) z_{t-1} + h$$

where $D_t = \text{diag}(z_t > 0)$.

**shPLRNN**: $z_t = A z_{t-1} + W_1 \Phi(W_2 z_{t-1} + h_2) + h_1$

**ALRNN**: Only last $P$ dimensions use ReLU; $d_{1:M-P} = 1 \; \forall t$.

## TDD Tests — `tests/test_models.py`

| Test | Assertion |
|------|-----------|
| `test_plrnn_forward_shape` | `forward(z)` returns shape `(M,)` / `(B, M)` |
| `test_plrnn_diagonal_A` | `A` is stored as a diagonal vector, applied element-wise |
| `test_plrnn_jacobian` | `get_jacobian(z) == A_diag + W·D(z)`, verified against finite-diff |
| `test_plrnn_D_computation` | `D(z)` is binary diagonal: `d_i = 1 iff z_i > 0` |
| `test_plrnn_subregion_id` | Consistent hashable ID from sign pattern of `z` |
| `test_plrnn_subregion_count` | Base PLRNN: `2^M` possible regions |
| `test_shplrnn_shapes` | `W1` is `(M,H)`, `W2` is `(H,M)`, `h2` is `(H,)` |
| `test_shplrnn_jacobian` | `J = diag(A) + W1·D_h·W2` where `D_h = diag(W2·z + h2 > 0)` |
| `test_alrnn_linear_dims` | First `M-P` components always pass linearly (`d_i = 1`) |
| `test_alrnn_subregion_count` | `2^P` possible regions (not `2^M`) |
| `test_trajectory_shape` | `forward_trajectory(z0, T)` returns `(T+1, M)` |
| `test_trajectory_deterministic` | Same z0 → same trajectory |

## Files

### [NEW] `src/dynamic/models/__init__.py`

Exports: `PLRNN`, `ShallowPLRNN`, `ALRNN`

### [NEW] `src/dynamic/models/plrnn.py`

```python
class PLRNN(nn.Module):
    """Base PLRNN: z_t = diag(A) z_{t-1} + W ReLU(z_{t-1}) + h"""

    def __init__(self, M: int)
    def forward(self, z: Tensor) -> Tensor
    def get_D(self, z: Tensor) -> Tensor  # diagonal ReLU mask
    def get_jacobian(self, z: Tensor) -> Tensor  # A + W·D(z)
    def get_subregion_id(self, z: Tensor) -> tuple[int, ...]
    def forward_trajectory(self, z0: Tensor, T: int) -> Tensor
```

### [NEW] `src/dynamic/models/shallow_plrnn.py`

```python
class ShallowPLRNN(nn.Module):
    """shPLRNN with hidden layer H."""

    def __init__(self, M: int, H: int)
    def forward(self, z: Tensor) -> Tensor
    def get_D_hidden(self, z: Tensor) -> Tensor  # D_h from hidden activation
    def get_jacobian(self, z: Tensor) -> Tensor  # diag(A) + W1·D_h·W2
    def get_subregion_id(self, z: Tensor) -> tuple[int, ...]
    def forward_trajectory(self, z0: Tensor, T: int) -> Tensor
```

### [NEW] `src/dynamic/models/alrnn.py`

```python
class ALRNN(nn.Module):
    """Almost-Linear RNN: only last P dims use ReLU."""

    def __init__(self, M: int, P: int)
    def forward(self, z: Tensor) -> Tensor
    def get_D(self, z: Tensor) -> Tensor  # first M-P always 1
    def get_jacobian(self, z: Tensor) -> Tensor
    def get_subregion_id(self, z: Tensor) -> tuple[int, ...]
    def forward_trajectory(self, z0: Tensor, T: int) -> Tensor
```

## Verification

```bash
source .venv/bin/activate
python -m pytest tests/test_models.py -v
ruff check src/dynamic/models/
```
