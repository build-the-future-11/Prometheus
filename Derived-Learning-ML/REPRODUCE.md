# Reproduction guide

## Scope

These commands reproduce the package's deterministic CPU execution. They are an internal rerun recipe, not evidence that an independent party has reproduced the result.

Requirements: Python 3.11+, a POSIX-like shell, and no third-party Python packages. Run from this directory. Set `PYTHONHASHSEED=0` for a fully explicit process environment, although all experiment RNG seeds are derived with SHA-256 rather than Python's randomized hash.

Optional packaging check (offline, using the repository-owned build backend):

```bash
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/dlml-wheel .
```

## 1. Setup and tests

```bash
export PYTHONPATH="$PWD/src"
export PYTHONHASHSEED=0
python3 --version
python3 -m unittest discover -s tests -v
```

Expected: 17 tests pass. The suite includes an analytic-versus-finite-difference MSE-gradient check, an independent ridge normal-equation check, shape/batching/non-finite rejection, gradient isolation, a proof-by-monkeypatch that the proposed rule does not call the global-gradient function, model serialization, exact Adam-state resume, deterministic split generation, and an end-to-end smoke run.

## 2. Materialize data

```bash
python3 -m derived_learning generate \
  --config configs/development.json \
  --output runs/reproduction-tasks.json
```

The output includes separate `meta_train`, `development`, `final`, and `long_horizon` tasks. Hidden coefficient vectors are not serialized.

## 3. Run selection, training, and locked checkpoint evaluation

```bash
python3 -m derived_learning run \
  --config configs/development.json \
  --output-dir runs/reproduction \
  --project-root .
```

This single guarded command prevents a caller from injecting final results into selection. It writes:

- `tasks.json`: exact task data and provenance;
- `selection_attempts.jsonl`: every meta-training/development candidate attempt, including failures;
- `selection.json`: locked hyperparameters and selection rationale;
- `results.jsonl`: every final/long-horizon task, condition, checkpoint, weight vector, and resource counter;
- `summary.json`: paired equal-resource and quality-threshold analysis; and
- `manifest.json`: configuration/source identity, split membership, environment, resource scope, and hashes for all core outputs.

## 4. Independently recompute terminal evaluation metrics

```bash
python3 -m derived_learning evaluate \
  --run-dir runs/reproduction \
  --output runs/reproduction/evaluation_verification.json
```

Expected: `VERIFIED` with 32 checked results. This reloads task evaluation data and saved final weights, reconstructs each model, and rejects any terminal NMSE mismatch.

## 5. Generate the report from saved outputs

```bash
python3 -m derived_learning report \
  --run-dir runs/reproduction \
  --output runs/reproduction/REPORT.md
```

No numeric table entry is typed by hand; all are read from `summary.json`.

## 6. Verify manifest-bound artifacts

```bash
python3 -m derived_learning verify \
  --run-dir runs/reproduction \
  --output runs/reproduction/manifest_verification.json
```

Expected: `VERIFIED` with five checked core outputs. The verification receipt is intentionally outside the manifest it verifies.

## Determinism check

Run step 3 twice into two fresh directories, then compare the five core files listed in each manifest. `tasks.json`, `selection.json`, `selection_attempts.jsonl`, `results.jsonl`, and `summary.json` must be byte-identical on the same Python implementation. The manifest additionally records environment and Git identity and may therefore differ across machines or commits.

## S3 Kaggle source qualification

S3 is a frozen successor and does not change the CPU findings above. Build the
hash-bound private source artifact, then test the exact extracted payload:

```bash
bash scripts/build_s3_source_bundle.sh
audit_dir="$(mktemp -d)"
tar -xzf releases/s3-r0/dlml-s3-source.bundle -C "$audit_dir"
(
  cd "$audit_dir"
  PYTHONPATH=src PYTHONHASHSEED=0 python3 -m unittest discover -s tests -q
  python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
)
```

The local suite includes a generated complete hosted-evidence fixture that the
independent verifier must accept. The two release-only contract tests are
expected to skip inside the extracted payload because the nested payload does
not contain its own outer release archive or Kaggle metadata.

After Kaggle completes, download both `kaggle-dlml-s3-r0-r1.tar.gz` and its
`.sha256` sidecar and run:

```bash
python3 scripts/verify_s3_hosted_archive.py \
  --archive /path/to/kaggle-dlml-s3-r0-r1.tar.gz \
  --sidecar /path/to/kaggle-dlml-s3-r0-r1.tar.gz.sha256 \
  --config configs/s3_real_data_gpu.json \
  --project-root . \
  --output /path/to/s3-hosted-verification.json
```

Only `verification_status: VERIFIED` admits the recorded `registered_result`.
`SUPPORTED`, `MIXED`, `NEGATIVE`, and `INCONCLUSIVE` are all valid frozen
outcomes; an `INVALID` verification receipt admits no scientific result.

## Expected admitted finding

The committed run should reproduce a final split interpretation of `SUPPORTED_ON_ADMITTED_MATRIX` and a long-horizon interpretation of `NOT_SUPPORTED_ON_ADMITTED_MATRIX`. Treat any discrepancy as a reproduction failure to investigate; do not tune or replace seeds to recover the expected sign.

