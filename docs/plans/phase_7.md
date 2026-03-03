# Phase 7: Experiments & Visualization

**Status**: ✅ Complete
**Depends on**: Phase 3 (training), Phase 5 (manifolds), Phase 6 (analysis)
**Estimated effort**: Large

## Objective

Reproduce the paper's figures and validate against Table 2. Each experiment script runs the full pipeline: generate data → train model → detect fixed points → compute manifolds → evaluate metrics → generate figures.

## Experiments

### Fig 2: Invertibility Regularization Ablation

**Script**: `experiments/fig2_invertibility.py`

- Train ALRNN on Lorenz-63 with varying M (10, 20, 30, 50) and λ_invert ∈ {0, 0.1·exp(M)}
- Measure: proportion of subregions with det(J) > 0, reconstruction quality (D_stsp)
- Also: convergence comparison on damped oscillator with/without regularization

**Success**: Regularization achieves > 95% invertibility with minimal D_stsp impact.

### Fig 3: Toy PL Map Validation

**Script**: `experiments/fig3_toy_validation.py`

- **No training** — use PL map directly as PLRNN
- Fig 3A: Compute stable/unstable manifolds of saddle, verify forward/backward iterates lie on them
- Fig 3B: Trace stable manifold of period-4 and period-3 saddles (fig3b_left/right), delineate basins of attraction via `plot_basins_2d`

**Success**: Δ_σ ≈ 0.98 (Table 2, row 1).

### Fig 4A: Duffing Basin Boundaries

**Script**: `experiments/fig4a_duffing.py`

- Train shPLRNN (M=2, H=10) on Duffing data
- Find fixed points with SCYFI (two stable, one saddle)
- Compute stable manifold of saddle → basin boundary

**Success**: Δ_σ ≈ 0.97 (Table 2, row 2).

### Fig 4B: Decision-Making Multistability

**Script**: `experiments/fig4b_decision.py`

- Train ALRNN (M=15, P=6) on decision-making task data
- Find two stable FPs and the saddle between them
- Compute stable manifold → basin boundary in 15D (visualize in 3D subspace)

**Success**: Δ_σ ≈ 0.95 (Table 2, row 3).

### Fig 4C: Lorenz-63 Manifolds

**Script**: `experiments/fig4c_lorenz.py`

- Train shPLRNN (M=3, H=20) on Lorenz data
- Find saddle at bottom of attractor
- Compute stable and unstable manifolds
- Compare with numerical continuation of original ODE

**Success**: Δ_σ ≈ 0.78 (Table 2, row 4).

### Fig 5: Homoclinic Chaos

**Script**: `experiments/fig5_chaos.py`

- Use PL map with Fig 5 parameters (direct, no training)
- Compute stable and unstable manifolds of saddle
- Detect homoclinic intersections
- Generate bifurcation diagram varying h₁
- Compute Lyapunov exponents across chaotic range

**Success**: Homoclinic point detected. Positive max Lyapunov exponent confirming chaos.

## Visualization — `src/dynamic/viz/plotting.py`

| Function | Used in | Description |
|----------|---------|-------------|
| `plot_state_space_2d` | Fig 3, 4A, 5 | Trajectories + fixed points + manifold segments |
| `plot_state_space_3d` | Fig 4B, 4C | 3D subspace projection |
| `plot_basins_2d` | Fig 3B, 4A | Color-coded basins of attraction |
| `plot_manifold_quality` | Fig 6 (Appx) | δ_σ histograms: on-manifold vs off-manifold |
| `plot_bifurcation` | Fig 5C | Attractor values vs parameter |
| `plot_lyapunov_spectrum` | Fig 5D | Lyapunov exponents vs parameter |
| `plot_invertibility` | Fig 2 | Invertibility proportion vs dimension |

## Verification

```bash
# Run each experiment
source .venv/bin/activate
python experiments/fig3_toy_validation.py
python experiments/fig4a_duffing.py
python experiments/fig4b_decision.py
python experiments/fig4c_lorenz.py
python experiments/fig5_chaos.py
python experiments/fig2_invertibility.py

# Check output figures in experiments/output/
# Validate Δ_σ values against Table 2

# Linting
ruff check src/ tests/ experiments/
```
