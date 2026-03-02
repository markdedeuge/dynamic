# Phase 4: SCYFI — Fixed Point & Cycle Detection 🔬

**Status**: Not started — requires research phase
**Depends on**: Phase 1 (models)
**Estimated effort**: Large

## Objective

Port the SCYFI algorithm from Julia to Python/PyTorch. SCYFI (Searcher for CYcles and FIxed points) finds all fixed points and k-cycles in ReLU-based RNNs by exploiting their piecewise-linear structure.

## Research Prerequisites

Before implementation, complete the following:

1. **Download the SCYFI paper**: "Bifurcations and loss jumps in RNN training" (Eisenmann, Monfared, Göring, Durstewitz — NeurIPS 2023)
2. **Study the Julia source**: [github.com/DurstewitzLab/SCYFI](https://github.com/DurstewitzLab/SCYFI)
   - `src/` — core algorithm files
   - `test/` — test cases
   - `main.jl` — entry point showing API
3. **Document algorithm internals** in a SCYFI-specific sub-plan

### Known API (from Julia source)

```julia
# Standard PLRNN
cycles, eigenvalues = find_cycles(A, W, h, required_order;
    outer_loop_iterations=100, inner_loop_iterations=500)

# Shallow PLRNN
FPs, eigenvals = find_cycles(A, W₁, W₂, h₁, h₂, required_order;
    outer_loop_iterations=10, inner_loop_iterations=60)
```

### Algorithm Sketch

For **fixed points** (order 1):
1. For each possible subregion D (binary diagonal), solve `z* = (A + WD)z* + h` → `z* = (I - A - WD)⁻¹ h`
2. Check self-consistency: the signs of `z*` must match `D`
3. If consistent, `z*` is a valid fixed point

For **k-cycles** (order k):
1. Enumerate valid sequences of k subregions `(D₁, D₂, ..., D_k)`
2. For each sequence, compose the k affine maps and solve for the fixed point of `F^k`
3. Check self-consistency across all k states in the cycle

The key insight: PLRNNs are linear within each subregion, so fixed point equations become linear systems. SCYFI's efficiency comes from pruning infeasible subregion sequences.

## TDD Tests — `tests/test_scyfi.py`

| Test | Assertion |
|------|-----------|
| `test_find_fp_2d_pl_map_3a_left` | Finds known FP, ‖F(z*)-z*‖ < 1e-10 |
| `test_find_fp_2d_pl_map_3a_right` | Finds known FP for second param set |
| `test_fp_self_consistency` | Solved z* has signs matching the D used to solve |
| `test_find_period3_cycle` | Finds 3-cycle in Fig 3B right params |
| `test_find_period4_cycle` | Finds 4-cycle in Fig 3B left params |
| `test_cycle_is_periodic` | F^k(z*) = z* for all points in k-cycle |
| `test_cycle_points_distinct` | All k points in a k-cycle are distinct |
| `test_classify_saddle` | Mixed eigenvalues (some |λ|<1, some |λ|>1) → saddle |
| `test_classify_stable` | All |λ| < 1 → stable |
| `test_classify_unstable` | All |λ| > 1 → unstable |
| `test_eigenstructure` | Eigenvalues match `numpy.linalg.eig` of Jacobian |
| `test_shplrnn_support` | Works with shPLRNN parameters |
| `test_no_spurious_fps` | Doesn't return FPs that fail self-consistency |

## Files

### [NEW] `src/dynamic/analysis/subregions.py`

```python
def get_D(z: Tensor) -> Tensor
    """Diagonal ReLU activation matrix."""

def get_region_id(z: Tensor) -> tuple[int, ...]
    """Hashable subregion identifier from sign pattern."""

def get_neighbors(region_id: tuple) -> list[tuple]
    """Adjacent regions (single bit-flip in D)."""

def get_jacobian_in_region(model, D: Tensor) -> Tensor
    """Compute A + W·D for a given activation pattern."""

def classify_point(eigenvalues: ndarray) -> str
    """'stable' | 'unstable' | 'saddle' based on eigenvalue magnitudes."""
```

### [NEW] `src/dynamic/analysis/scyfi.py`

```python
def find_fixed_points(model) -> list[FixedPoint]
    """Find all fixed points by enumerating subregions."""

def find_cycles(model, max_order: int, **kwargs) -> list[Cycle]
    """Find all cycles up to given order."""

def analyze_eigenstructure(model, point: Tensor) -> EigenInfo
    """Eigenvalues, eigenvectors, and classification at a point."""

@dataclass
class FixedPoint:
    z: Tensor
    eigenvalues: ndarray
    eigenvectors: ndarray
    classification: str  # 'stable' | 'unstable' | 'saddle'
    region_id: tuple

@dataclass
class Cycle:
    points: list[Tensor]
    period: int
    eigenvalues: ndarray
    classification: str
    region_ids: list[tuple]
```

## Verification

```bash
python -m pytest tests/test_scyfi.py tests/test_subregions.py -v
ruff check src/dynamic/analysis/
```
