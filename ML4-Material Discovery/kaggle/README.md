# Kaggle execution

The latest completed study is `robust-aggregation-v3`, run
`kaggle-robust-aggregation-v3-r1`. Its launcher is
`robust/run_hosted_robust.py`; completed execution and downloaded evidence are
recorded in `artifacts/hosted-submission-v3.json` and
`artifacts/hosted-import-v3.json`. All 20 scientific cells completed, with 58
tests passing. `scripts/verify_evidence.py` recomputes the numerical evidence.
The exact frozen uploaded source is retained in
`releases/20261021/robust-source.bundle`, with its SHA-256 file. It is a tar
archive under an opaque extension to preserve the compressed data bytes on
Kaggle. The current release adds audit/documentation work and lazy plotting
imports; the frozen training source is retained separately.

For reproduction, assign a new immutable run ID and rebuild/freeze the source
bundle. Existing run output must never be overwritten. The primary aggregation
comparison is supported, but the overall study has mixed development findings;
see the root README and RESULTS for the exact scope.

To stage (but not submit) an exact replay of the retained v3 source, use a new
identity and a fresh output directory:

```bash
.venv/bin/python scripts/stage_kaggle_replay.py \
  --run-id kaggle-robust-aggregation-v3-replay-r2 \
  --output /tmp/ml4-v3-replay-r2
```

The emitted private kernel metadata binds the replay to the retained source
bundle hash. Review `stage-manifest.json`, then explicitly submit with the
command it records. This staging route cannot change the model or protocol.

## Preserved first-study launcher

The Kaggle artifact consists of a private source dataset and a private script
notebook. `kernel/run_hosted.py` copies the read-only dataset to Kaggle working
storage and invokes `scripts/run_kaggle.sh` with the immutable run ID
`kaggle-development-v1-r1`.

Kaggle transparently expands recognized compressed dataset files. The original
SHA-256-bound `.json.gz` input is therefore also transported inside the opaque
`frozen-data.bundle`; the launcher restores that exact byte sequence before the
data contract and integration tests execute.

The runner executes the test suite, validates the pinned data, records
preflight admission, runs the unchanged `development-v1` protocol, verifies
the run manifest, enforces the historical artifact-completeness gate, and emits a
SHA-256-bound archive. A failed resource guard or scientific run remains a
failed Kaggle version; it is never converted into a successful result.
The historical gate alone is not numerical or scientific verification.
