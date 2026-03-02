# Phase 0: Dependencies & Project Skeleton

**Status**: ✅ Complete
**Depends on**: Nothing
**Estimated effort**: Small

## Objective

Set up the full project directory structure and install all remaining dependencies needed for the implementation.

## Changes

### [MODIFY] pyproject.toml

Add to `dependencies`:
- `scipy` — ODE integration for benchmark systems
- `scikit-learn` — PCA, kernel-PCA for manifold fitting
- `matplotlib` — Plotting and figure generation

Add to `[project.optional-dependencies] dev`:
- `hdbscan` — Clustering for Algorithm 3 fallback

### [NEW] Package directories

Create `__init__.py` files for:
- `src/dynamic/models/`
- `src/dynamic/systems/`
- `src/dynamic/training/`
- `src/dynamic/analysis/`
- `src/dynamic/viz/`

## Verification

```bash
uv sync --all-extras
source .venv/bin/activate
python -c "import scipy, sklearn, matplotlib, hdbscan; print('All deps OK')"
python -m pytest tests/ -v  # Existing tests still pass
ruff check src/ tests/
```
