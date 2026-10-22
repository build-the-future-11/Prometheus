# ML4-Material Discovery

A fully implemented, resource-bounded study of composition-only experimental
band-gap prediction, uncertainty, and retrospective candidate selection.
Read `RESEARCH_STATE.json` first. Current results are in `RESULTS.md`.

## Verified result

The grouped v3 study completed 15 outer cells and five element holdouts on
Kaggle. Median aggregation of the same 288 trees reduced mean MAE from
**0.473870 to 0.425622 eV** across three predefined repetitions, satisfying the
0.01 eV improvement threshold in each. Ridge: 0.815014 eV; random forest:
0.519949 eV; histogram boosting: 0.548780 eV. The primary development hypothesis
is **SUPPORTED**. The overall scientific scope remains **MIXED**: positive-gap
error and RMSE worsened, several element-holdout intervals remain miscalibrated,
and there is no independent or prospective validation.

Hosted verification: **58 tests passed**, seven-cell synthetic end-to-end check,
424.3-second scientific run, 372.9 MiB peak RSS, zero pageouts, exact checkpoint
reloads, and independent raw-data/metric verification. These are real retained
Kaggle outputs, not fixture results. Historical official v1 and negative v2
results and every failed local run are preserved.

## Files to read

- `RESEARCH_STATE.json`: canonical contract, decisions, history and release state.
- `RESULTS.md`: numerical results and scope.
- `FLAW_AUDIT.md`: 38 concrete flaws, repairs and evidence requirements.
- `LITERATURE_AUDIT.md`: primary-source prior art and novelty limits.
- `PROTOCOLS/robust-aggregation-v3.md`: frozen successor specification.
- `paper/main.tex` and `paper/main.pdf`: complete paper using the recovered style.
- `paper/generated/evidence-map.json`: claims, figures and source hashes.
- `runs/kaggle-robust-aggregation-v3-r1/`: raw predictions, splits, models, metrics.
- `artifacts/hosted-v3/`: Kaggle test, dry-run, environment and execution logs.
- `history/pre-closeout-20261021/`: superseded documents, retained explicitly.

Post-release hardening adds 72 passing software tests, explicit matrix and subgroup
checks, independent bootstrap/random-ranking recomputation, and refusal to overwrite
frozen provenance. See `artifacts/adversarial-audit-20261021.json` and
`artifacts/resampling-audit-v3-20261021.json`.

A fresh Kaggle execution of the exact original frozen source reproduces all 20
cells: prediction files, splits and all 20 checkpoints are byte-identical.
Scientific summaries agree exactly excluding timing. See
`artifacts/reproduction-comparison-20261021.json`. This is same-operator
computational reproduction; independent scientific replication remains absent.

## Install and inspect

Use Python 3.12 or 3.13. The recorded Kaggle core versions are in
`requirements-hosted.txt`; the local `uv.lock` is a distinct build environment.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-hosted.txt -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_evidence.py --run-id kaggle-robust-aggregation-v3-r1
.venv/bin/python -m ml4materials predict-robust \
  --checkpoint runs/kaggle-robust-aggregation-v3-r1/models/repeat_0_fold_0.joblib \
  --formula GaAs --formula ZnO
```

The inference command uses a saved development fold. It returns eV predictions,
90% intervals and unseen-element flags; it is not a production certification.
Only deserialize trusted project checkpoints.

## Reproduce the science

Run on a stable CPU host, preferably Kaggle, with numerical-library thread
counts fixed to one. Keep the 512 MiB/no-rising-pageout guards unchanged.
Use a new run ID; existing runs are immutable.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/dry_run_robust.py
.venv/bin/python -m ml4materials run-robust --run-id reproduction-v3-r1
.venv/bin/python scripts/verify_evidence.py --run-id reproduction-v3-r1
.venv/bin/python -m scripts.generate_paper_assets --robust-run reproduction-v3-r1
tectonic paper/main.tex
```

The Kaggle launcher is `kaggle/robust/run_hosted_robust.py`; its exact completed
run is linked in `artifacts/hosted-submission-v3.json`. The launcher keeps failed
outputs and checks dependencies before training. Do not resubmit it unchanged:
assign a new run ID and freeze the source inventory for a reproduction.

## Interpretation

The original official splits have canonical overlaps. The successor averages
three conflicting duplicate groups and uses grouped splits, so its scores are
not official leaderboard scores. All Matbench labels were previously inspected;
new split seeds do not make them a protected test set. Neural composition
baselines and external validation were not executed. No SOTA, stability,
synthesizability, toxicity, novelty, device, or discovery claim is made.
