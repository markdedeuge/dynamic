# Phase 2: Benchmark Dynamical Systems

**Status**: Not started
**Depends on**: Phase 0
**Estimated effort**: Medium

## Objective

Implement the 5 benchmark dynamical systems used in the paper as data generators. Each system provides training trajectories via ODE integration or direct map iteration.

## Systems

### 1. 2D Piecewise-Linear Map (Gardini et al.)

From Appendix H.1, Eq 48:

$$F(X) = \begin{cases} A_l \cdot X + B, & \text{if } x \leq 0 \\ A_r \cdot X + B, & \text{if } x \geq 0 \end{cases}$$

PLRNN reformulation: $A = A_l$, $W = \begin{pmatrix} \tau_r - \tau_l & 0 \\ -\delta_r + \delta_l & 0 \end{pmatrix}$, $h = B$.

**Preset parameter configs**:
- **Fig 3A left**: τ_r=1.2, δ_r=1.8, τ_l=-0.3, δ_l=0.9, c=-1.5, d=1.0, h₂=-0.1, h₁=-0.13
- **Fig 3A right**: τ_r=1.26, δ_r=0.71, τ_l=-0.39, δ_l=-0.91, c=-0.44, d=0.56, h₂=0.62, h₁=-0.28
- **Fig 3B left**: τ_r=-1.85, δ_r=0.9, τ_l=-0.3, δ_l=0.9, c=1, d=0, h₂=0, h₁=-1
- **Fig 3B right**: τ_r=-1.85, δ_r=0.9, τ_l=0.9, δ_l=0.9, c=1, d=0, h₂=0, h₁=-1
- **Fig 5 (chaos)**: τ_r=1.5, δ_r=0.75, τ_l=-1.77, δ_l=0.9, c=0.6, d=0.15, h₂=-0.4, h₁=-0.7

### 2. Duffing Oscillator (Appendix H.3)

$$\ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 = \gamma \cos(\omega t)$$

Parameters: α=-1, β=0.1, δ=0.5, γ=0 (unforced, bistable regime).

### 3. Lorenz-63 (Appendix H.3)

$$\dot{x}_1 = \sigma(x_2 - x_1), \quad \dot{x}_2 = x_1(\rho - x_3) - x_2, \quad \dot{x}_3 = x_1 x_2 - \beta x_3$$

Parameters: σ=10, ρ=28, β=8/3.

### 4. Damped Nonlinear Oscillator (Appendix H.3)

10D system: $\dot{x}_i = y_i$, $\dot{y}_i = -\alpha_i x_i - \beta_i x_i^3 - \gamma_i y_i$

Explicit parameter vectors from the paper.

### 5. Multistable Decision-Making Model (Gerstner et al.)

3-variable ODE with sigmoid `g_E(h; θ) = 1/(1 + exp(-(h-θ)))`.

Parameters: τ_E=10, τ_inh=5, w_EE=16, w_EI=-15, w_IE=12, R=1, I₁=I₂=40, γ=1, θ=5.

## TDD Tests — `tests/test_systems.py`

| Test | Assertion |
|------|-----------|
| `test_pl_map_fixed_points` | Analytical FPs match for each parameter set |
| `test_pl_map_as_plrnn` | PL map → PLRNN reformulation gives identical output |
| `test_pl_map_iterate` | 1000-step trajectory stays bounded |
| `test_duffing_two_equilibria` | Two stable FPs exist for bistable params |
| `test_duffing_trajectories_bounded` | Long trajectories don't diverge |
| `test_lorenz_three_fixed_points` | FPs at origin and (±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1) |
| `test_lorenz_attractor_both_lobes` | Trajectory visits both wings of butterfly |
| `test_lorenz_bounded` | 10000-step trajectory stays bounded |
| `test_oscillator_energy_decreasing` | Total energy is non-increasing over time |
| `test_oscillator_dimension` | 20D state (10 position + 10 velocity) |
| `test_decision_two_stable_states` | Two distinct stable steady states from different ICs |
| `test_decision_sigmoid_correct` | g_E matches known sigmoid values |

## Files

### [NEW] `src/dynamic/systems/pl_map.py`

```python
class PLMap:
    """2D piecewise-linear map from Gardini et al."""
    def __init__(self, tau_r, delta_r, tau_l, delta_l, c, d, h1, h2)
    def step(self, X: ndarray) -> ndarray
    def trajectory(self, X0: ndarray, T: int) -> ndarray
    def to_plrnn_params(self) -> dict  # A, W, h matrices
    def analytical_fixed_points(self) -> list[ndarray]

    @classmethod
    def fig3a_left(cls) -> PLMap
    @classmethod
    def fig3a_right(cls) -> PLMap
    # ... preset configs for each figure
```

### [NEW] `src/dynamic/systems/duffing.py`, `lorenz63.py`, `oscillator.py`, `decision.py`

Each provides:
```python
def generate_trajectory(x0, T, dt=0.01) -> ndarray
def ode_rhs(t, x) -> ndarray
```

## Verification

```bash
python -m pytest tests/test_systems.py -v
ruff check src/dynamic/systems/
```
