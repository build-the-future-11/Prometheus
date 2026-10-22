# Reproduction guide

## Environment

The lockfile is authoritative. Python 3.12 is selected to avoid relying on the host's newer interpreter.

```bash
export UV_CACHE_DIR=/tmp/ml4-material-uv-cache
uv sync --extra dev --python 3.12 --frozen
```

## Data verification

The raw dataset, complete upstream validation bundle, and metadata are retained under `data/raw/`. No API key is needed.

```bash
uv run ml4materials prepare
shasum -a 256 data/raw/matbench_expt_gap.json.gz
shasum -a 256 data/raw/matbench_v0.1_validation.json
shasum -a 256 data/raw/matminer_dataset_metadata.json
```

Expected hashes are in `configs/data_contract.json`.

## Tests

```bash
uv run pytest -q
```

## Development experiment

Run IDs must be new. Never overwrite or tune a failed run into success.

```bash
/usr/bin/time -l uv run ml4materials run --run-id development-v1-r3
uv run ml4materials verify --run-id development-v1-r3
```

The run retains source predictions, a unique-composition candidate ledger, fold checkpoints, tables, figures, failures, resource receipt, and an SHA-256 manifest. Root `RESULTS.md` and `MANUSCRIPT.md` are results-derived copies.

## Successor experiment lifecycle

Plan and preflight without allocating a run directory:

```bash
.venv/bin/python -m ml4materials plan --config configs/studies/quantile-composition-v2.json
.venv/bin/python -m ml4materials preflight --output artifacts/preflight-$(date +%Y%m%d).json
```

The direct-quantile successor is a separate study and uses a fresh ID:

```bash
.venv/bin/python -m ml4materials run-quantile --run-id quantile-composition-v2-r2
```

Index retained receipts or compare only completed runs with identical input hashes:

```bash
.venv/bin/python -m ml4materials registry --output artifacts/run_registry.json
.venv/bin/python -m ml4materials compare --run-id run-a --run-id run-b
```

Before a numerical paper export, the release gate refuses incomplete, changed, or unmanifested runs:

```bash
.venv/bin/python scripts/reproduce_submission.py --run-id <completed-run-id>
```

## Inference

Checkpoint predictions are fold-model predictions. They are suitable for interface verification, not a separately trained production model.

```bash
uv run ml4materials predict \
  --checkpoint runs/development-v1-r3/models/fold_0.joblib \
  --formula GaAs --formula ZnO
```

For a CSV with `candidate_id,composition`:

```bash
uv run ml4materials rank-csv \
  --checkpoint runs/development-v1-r3/models/fold_0.joblib \
  --input candidates.csv \
  --output ranked_candidates.csv
```

## Status interpretation

Passing tests means the local software contract held. A successful run means the numerical workflow executed. Neither means publication readiness, prospective discovery, deployment certification, or independent replication.

