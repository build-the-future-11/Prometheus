# Derived Learning ML

This directory is a dependency-free, CPU-reproducible research package for one bounded study of analytically derived learning targets. It began as an empty child directory on 2026-10-19, so `PROJECT_CHARTER.md` records the required provisional hypothesis rather than attributing an earlier mechanism to the folder name.

## Implemented study

DLML-S1-P0 tests a grouped additive linear model. The primary method derives a target for each block by subtracting the other blocks' current predictions from the support label, then solves the block with ridge regression. It is compared on paired tasks with:

- a development-selected full-batch Adam baseline;
- a fixed normalized-LMS rule; and
- an ordinary block-gradient ablation that uses the same block schedule without derived targets.

No learned target generator, teacher, component-label supervision, remote service, or GPU is used. Final and long-horizon tasks are locked out of hyperparameter selection. The package retains every candidate/task attempt and separately accounts for arithmetic work, support reads, global-gradient calls, teacher calls, probes, preprocessing, and target-generator training.

## Current evidence

The deterministic smoke and admitted development runs are saved under `runs/`. The five-seed final family passed both predeclared criteria: the primary method's equal-resource mean NMSE was 0.003475 versus Adam's 0.024140, and median work to NMSE 0.08 was 10,816 versus 41,184 units. The three-seed longer-horizon family failed the quality criterion: its primary equal-resource NMSE was 1.13% worse than Adam, although it reached the threshold sooner. See the generated `RESULTS.md` and machine-readable `runs/development/summary.json`.

These are internal synthetic results, not independent reproduction or evidence of general neural-learning superiority. The primary update is closely related to classical partial-residual backfitting/block coordinate minimization; `RELATED_WORK.md` makes that boundary explicit.

## Quick start

Python 3.11 or newer is the only runtime requirement. The repository includes a minimal PEP 517 backend, so wheel builds do not download setuptools or any other build dependency.

```bash
export PYTHONPATH="$PWD/src"
python3 -m unittest discover -s tests -v
python3 -m derived_learning generate \
  --config configs/smoke.json \
  --output runs/local-smoke-tasks.json
python3 -m derived_learning run \
  --config configs/smoke.json \
  --output-dir runs/local-smoke \
  --project-root .
python3 -m derived_learning evaluate \
  --run-dir runs/local-smoke \
  --output runs/local-smoke/evaluation_verification.json
python3 -m derived_learning report \
  --run-dir runs/local-smoke \
  --output runs/local-smoke/REPORT.md
python3 -m derived_learning verify \
  --run-dir runs/local-smoke \
  --output runs/local-smoke/manifest_verification.json
```

An optional offline wheel build is also supported:

```bash
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

## Frozen S3 Kaggle run

`DLML-S3-R0` is a separately frozen real-data GPU study on the public Kaggle
California Housing dataset. It compares residual-target and joint-training
additive block MLPs at equal example passes, plus ridge and histogram-gradient
boosting baselines, across five protected splits. Build and verify its private
source artifact with:

```bash
bash scripts/build_s3_source_bundle.sh
PYTHONPATH=src python3 -m unittest discover -s tests -q
```

The builder rejects unsafe paths, links, macOS AppleDouble files, and Finder
metadata before writing the SHA-256 sidecar. The private Kaggle launcher is
`kaggle/run_hosted_s3.py`; its typed metadata is
`kaggle/kernel-metadata.json`. After a hosted run, independently verify the
downloaded archive before changing any result or paper claim:

```bash
python3 scripts/verify_s3_hosted_archive.py \
  --archive /path/to/kaggle-dlml-s3-r0-r1.tar.gz \
  --sidecar /path/to/kaggle-dlml-s3-r0-r1.tar.gz.sha256 \
  --config configs/s3_real_data_gpu.json \
  --project-root . \
  --output /path/to/s3-hosted-verification.json
```

The verifier recomputes the result matrix and terminal classification from raw
records, checks the failure denominator and tracebacks, and refuses broad or
independent-reproduction claims. See `HOSTED_EXECUTION_STATUS.md` for the
current remote handoff state.

The `run` command intentionally performs selection, locked training, and checkpoint scoring as one guarded workflow. The separate `evaluate` command reconstructs every final model from saved weights and recomputes terminal evaluation NMSE as an independent readback.

## Repository map

- `PROJECT_CHARTER.md`: provisional S1 scope and mechanism taxonomy.
- `PROTOCOL.md`: frozen matrix, selection rule, resource model, and falsifier.
- `src/derived_learning/`: generator, model, update rules, orchestration, reporting, and verification.
- `configs/`: deterministic smoke and admitted development matrices.
- `tests/`: numerical, shape, isolation, stability, state, serialization, split, and end-to-end checks.
- `runs/`: materialized tasks, all attempts, results, summaries, manifests, and verification receipts.
- `RESULTS.md`: generated table; do not hand-edit its numeric values.
- `REPRODUCE.md`: exact clean-room command sequence and expected evidence.
- `WORK_QUEUE.md`: completed work and bounded successors.
- `scripts/build_s3_source_bundle.sh`: fail-closed S3 source packaging.
- `scripts/verify_s3_hosted_archive.py`: independent hosted-artifact verifier.
- `kaggle/`: private launcher and typed Kaggle kernel metadata.

