# Phase 5: Manifold Detection (Algorithms 1, 2, 3) 🔬

**Status**: Not started
**Depends on**: Phase 1 (models), Phase 4 (SCYFI for saddle points)
**Estimated effort**: Large — this is the paper's core contribution

## Objective

Implement the novel manifold detection algorithm from §3.3. This is the main contribution of the paper: a semi-analytical method for computing stable and unstable manifolds of saddle points in PLRNNs.

## Algorithm Overview

### Algorithm 1 — Manifold Construction (§3.3)

1. **Initialize**: At saddle point `p`, compute eigenvectors of the Jacobian `A + WD(p)`. Stable eigenspace (|λ| < 1) defines the local stable manifold; unstable eigenspace (|λ| > 1) defines the local unstable manifold.
2. **Sample**: Place N_s points on the local manifold using eigenvalue-rescaled GMM sampling (scales σ ∈ {0.01, 0.1, 0.5}, QR orthonormalization, rejection sampling).
3. **Propagate**: Push points forward (unstable) or backward (stable) into neighboring subregions.
4. **Fit**: In each new subregion, fit the manifold segment with PCA (planar) or kernel-PCA with RBF kernel (curved, when complex eigenvalues or defective eigenvalues present).
5. **Iterate**: Repeat for N_iter iterations, expanding across the state space.

### Algorithm 2 — Backtracking / Map Inversion (Appendix C)

For stable manifolds, we need `F_θ⁻¹`: solve `z_{t-1} = (A + WD_{t-1})⁻¹(z_t - h)`.

Self-consistency problem: the `D_{t-1}` on the RHS depends on the unknown `z_{t-1}` on the LHS.

Heuristic resolution:
1. Try backward step using current region's D
2. Verify via forward step: `F(z*) == z_t`?
3. If fail → update D from candidate solution, retry
4. If fail → try previously visited D pool
5. If fail → hierarchical bitflip search

### Algorithm 3 — Fallback (Appendix C)

For manifolds with folding/discontinuous structure:
- Perturb seed points, generate trajectories
- Cluster per subregion using HDBSCAN
- Fit PCA/kPCA per cluster

## TDD Tests

### `tests/test_backtracking.py`

| Test | Assertion |
|------|-----------|
| `test_backward_forward_consistency` | `F(F⁻¹(z)) == z` to < 1e-8 for random z |
| `test_backward_correct_region` | Backward step lands in the expected subregion |
| `test_bitflip_finds_neighbor` | When direct inversion fails, bitflip search succeeds |
| `test_backtrack_trajectory` | Full backward trajectory of length T is self-consistent |

### `tests/test_manifolds.py`

| Test | Assertion |
|------|-----------|
| `test_local_manifold_eigenvectors` | Local manifold at saddle is spanned by stable eigenvectors |
| `test_local_manifold_dimension` | Manifold dimension = number of stable eigenvalues (for stable manifold) |
| `test_sample_multi_scale` | Sampled points cover multiple scales (0.01, 0.1, 0.5) |
| `test_sample_rejection` | Points violating ReLU constraints are rejected |
| `test_pca_recovers_line` | PCA on collinear 2D points → 1 principal component |
| `test_kpca_captures_curve` | kPCA on spiral data → 1 curved component |
| `test_manifold_2d_toy_saddle` | Full Algo 1 on 2D PL map: δ_σ ≈ 0 for all manifold points |
| `test_manifold_spans_regions` | Manifold construction visits ≥ 2 subregions |
| `test_manifold_stable_unstable` | σ=+1 produces stable manifold, σ=-1 produces unstable |
| `test_fallback_on_folding` | Algo 3 fallback handles folded manifold structure |

## Files

### [NEW] `src/dynamic/analysis/backtracking.py`

```python
def backward_step(model, z_t: Tensor, D: Tensor) -> Tensor
    """Single backward step: z_{t-1} = (A + WD)⁻¹(z_t - h)"""

def verify_forward(model, z_prev: Tensor, z_target: Tensor, tol: float) -> bool
    """Check F(z_prev) ≈ z_target."""

def try_previous_regions(model, D_pool, z_t, z_candidate) -> Tensor | None
    """Try inversion using previously visited D matrices."""

def try_bitflips(model, z_t: Tensor, z_candidate: Tensor) -> Tensor | None
    """Hierarchical bitflip search over neighboring regions."""

def backtrack_trajectory(model, z_T: Tensor, T: int) -> Tensor
    """Full backward trajectory of length T."""
```

### [NEW] `src/dynamic/analysis/manifolds.py`

```python
@dataclass
class ManifoldSegment:
    region_id: tuple
    support_point: Tensor
    eigenvectors: Tensor  # or kPCA components
    is_curved: bool

def compute_local_manifold(model, saddle: FixedPoint, sigma: int) -> ManifoldSegment
    """Eigenvector decomposition at saddle, sigma=+1 stable, -1 unstable."""

def sample_on_manifold(segment: ManifoldSegment, N_s: int, c: float) -> Tensor
    """Eigenvalue-rescaled GMM sampling with rejection."""

def propagate_to_next_region(model, points: Tensor, sigma: int) -> dict[tuple, Tensor]
    """Push points forward/backward, group by destination subregion."""

def fit_manifold_segment(points: Tensor, eigenvalues: ndarray) -> ManifoldSegment
    """PCA (real eigenvalues) or kPCA (complex/defective)."""

def construct_manifold(
    model, saddle: FixedPoint, sigma: int,
    N_s: int = 100, N_iter: int = 50,
) -> list[ManifoldSegment]
    """Full Algorithm 1: iterative manifold construction."""
```

### [NEW] `src/dynamic/analysis/fallback.py`

```python
def fallback_manifold_detection(
    model, saddle: FixedPoint, sigma: int,
    N_forward: int = 100, N_backward: int = 100,
) -> list[ManifoldSegment]
    """Algorithm 3: trajectory perturbation + HDBSCAN clustering."""
```

## Verification

```bash
python -m pytest tests/test_backtracking.py tests/test_manifolds.py -v
ruff check src/dynamic/analysis/
```
