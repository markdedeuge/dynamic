# Phase 6: Analysis Tools

**Status**: ✅ Complete
**Depends on**: Phase 5 (manifold detection)
**Estimated effort**: Medium

## Objective

Implement the analysis tools required to evaluate manifold quality, detect chaos through homoclinic intersections, and compute Lyapunov exponents.

## Components

### 1. Manifold Quality Metric (Eq 6)

$$\delta_\sigma(x_0) := \frac{\min_{k\sigma \geq 0} \|F_\theta^k(x_0) - p\|_2^2}{\|x_0 - p\|_2^2} \in [0, 1]$$

Points on the manifold should have δ_σ ≈ 0 (they converge to `p`). The aggregate statistic is:

$$\Delta_\sigma := \langle I_U \rangle - \tilde{\delta}_\sigma(x_0 \in W^\sigma(p) \cap U)$$

where `I_U` is the indicator for points with δ_σ exceeding the max on-manifold δ_σ.

### 2. Homoclinic/Heteroclinic Detection

Intersections between stable and unstable manifolds:
- **2D analytical**: Algorithm 4 from Appendix I.2 (fold point method)
- **General**: Geometric intersection of manifold segments across subregions

### 3. Lyapunov Exponents

Compute from model Jacobians evolved along trajectories using QR decomposition method. Positive max Lyapunov exponent confirms chaos.

## TDD Tests

### `tests/test_quality.py`

| Test | Assertion |
|------|-----------|
| `test_delta_on_manifold_near_zero` | Points on known manifold → δ_σ < 0.01 |
| `test_delta_off_manifold_positive` | Random points far from manifold → δ_σ > 0.1 |
| `test_delta_bounded` | δ_σ ∈ [0, 1] for all points |
| `test_delta_statistic_positive` | Δ_σ > 0.5 when manifold is well-reconstructed |
| `test_delta_statistic_near_one` | 2D toy map → Δ_σ ≈ 0.98 |

### `tests/test_homoclinic.py`

| Test | Assertion |
|------|-----------|
| `test_analytical_homoclinic_fig5` | Detect homoclinic point in Fig 5 PL map params |
| `test_no_homoclinic_fig3a` | No homoclinic point in simple saddle case |
| `test_algo1_matches_analytical` | Algo 1 intersection detection agrees with analytical |
| `test_homoclinic_point_on_both` | Detected point lies on both W^{+1} and W^{-1} |

### `tests/test_lyapunov.py`

| Test | Assertion |
|------|-----------|
| `test_lyapunov_stable_fp` | Stable FP → all exponents < 0 |
| `test_lyapunov_chaos_fig5` | Chaotic PL map → max exponent > 0 |
| `test_lyapunov_sum_rule` | Sum of exponents relates to determinant product |
| `test_lyapunov_lorenz_positive` | Lorenz-63 → positive max exponent ≈ 0.9 |

## Files

### [NEW] `src/dynamic/analysis/quality.py`

```python
def delta_sigma(
    x0: Tensor, model, saddle: Tensor, sigma: int, k_max: int = 1000,
) -> float
    """Manifold convergence metric (Eq 6)."""

def delta_sigma_statistic(
    model, saddle: Tensor, sigma: int,
    U_min: Tensor, U_max: Tensor,
    N_samples: int = 1000, k_max: int = 500,
    manifold_points: Tensor | None = None,
) -> float
    """Full Δ_σ quality statistic."""
```

### [NEW] `src/dynamic/analysis/homoclinic.py`

```python
def find_homoclinic_intersections(
    stable_manifold: list[ManifoldSegment],
    unstable_manifold: list[ManifoldSegment],
) -> list[Tensor]
    """Find intersection points between stable and unstable manifolds."""

def analytical_homoclinic_2d(model, saddle, N_s: int, N_iter: int) -> list[Tensor]
    """Algorithm 4 (Appx I.2): analytical homoclinic detection for 2D PL maps."""
```

### [NEW] `src/dynamic/analysis/lyapunov.py`

```python
def compute_lyapunov_exponents(
    model, z0: Tensor, T: int = 10000,
) -> ndarray
    """Compute Lyapunov spectrum via QR decomposition of Jacobian products."""
```

## Verification

```bash
python -m pytest tests/test_quality.py tests/test_homoclinic.py tests/test_lyapunov.py -v
ruff check src/dynamic/analysis/
```
