# ML4-Material Discovery

This subproject is a narrow, evidence-bound study of **experimental inorganic band-gap prediction from composition** and **retrospective fixed-pool candidate selection**. The folder was empty at intake, so the user-authorized fallback objective was applied. It is not a universal materials model and does not claim a new material.

The primary dataset is the composition-only Matbench v0.1 `matbench_expt_gap` task (4,604 rows; target `gap expt`, eV). The model receives only formula-derived features available before measurement. Structures, relaxed geometries, calculated properties, database metadata, and post-synthesis information are excluded.

## What is implemented

- SHA-256-bound data loading and exact official Matbench five-fold manifests.
- A scale- and order-invariant composition parser and fixed feature schema.
- Train-median and standardized ridge baselines.
- A three-seed ExtraTrees ensemble with group-isolated calibration data.
- Normalized split-conformal intervals and uncertainty-driven abstention analysis.
- Fixed-pool random, greedy, uncertainty, and calibrated-probability ranking policies at equal budgets.
- Separately labeled Li, Zn, Ga, Se, and Te element-holdout studies.
- Label-permutation, checkpoint reload, split, duplicate, missing-data, and invariance checks.
- Durable predictions, candidate provenance, tables, figures, model checkpoints, failure logs, and hashes.

## Quick start

```bash
UV_CACHE_DIR=/tmp/ml4-material-uv-cache uv sync --extra dev --python 3.12
UV_CACHE_DIR=/tmp/ml4-material-uv-cache uv run ml4materials prepare
UV_CACHE_DIR=/tmp/ml4-material-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/ml4-material-uv-cache uv run ml4materials run --run-id development-v1-r3
UV_CACHE_DIR=/tmp/ml4-material-uv-cache uv run ml4materials verify --run-id development-v1-r3
```

After a run, predict formulas from one fold checkpoint:

```bash
UV_CACHE_DIR=/tmp/ml4-material-uv-cache uv run ml4materials predict \
  --checkpoint runs/development-v1-r3/models/fold_0.joblib \
  --formula GaAs --formula ZnO
```

Rank a CSV with a required `composition` column:

```bash
UV_CACHE_DIR=/tmp/ml4-material-uv-cache uv run ml4materials rank-csv \
  --checkpoint runs/development-v1-r3/models/fold_0.joblib \
  --input candidates.csv --output ranked_candidates.csv
```

See [PROTOCOL.md](PROTOCOL.md), [DATA_CARD.md](DATA_CARD.md), [MODEL_CARD.md](MODEL_CARD.md), [REPRODUCE.md](REPRODUCE.md), and the generated [RESULTS.md](RESULTS.md).

## Claim boundary

An executed benchmark is not prospective discovery. Ranked compositions are not verified stable, synthesizable, safe, useful, or novel. Official-split results are preserved for comparison even though an audit found three upstream canonical duplicate groups, two of which cross official train/test boundaries in four folds. Custom calibration and extrapolation splits keep canonical groups intact.

