# Reproducibility and evidence guide

This directory is a separately versioned successor to the closed v3 study. The frozen v4 claim is bounded by `SPECIFICATION.md`, `protocol.json`, `runs/freeze.json`, and the pre-confirmation Git identity in `runs/preconfirmation_anchor.json`. Do not revise the frozen method in place after inspecting confirmation or transfer outcomes.

## Evidence levels

1. **Engineering verification:** 26 unit tests and independent batch-solution checks establish declared implementation behavior.
2. **Development:** `runs/development-02` selected one candidate and one baseline from the registered 220-attempt matrix. It is selection evidence, not confirmation.
3. **Protected synthetic confirmation:** `runs/confirmation-02` contains the sole local 1,400-attempt registered campaign. `runs/analysis.json` applies the frozen decision rule.
4. **Chronological transfer:** UCI Air Quality is exposed regression evidence. Beijing PM2.5 is the preselected protected stream. Each is one correlated sequence, not an IID sample of datasets.
5. **Hosted replay:** `hosted/` contains a downloaded Kaggle archive and a local verification receipt when available. It demonstrates hosted portability, not independent external replication.
6. **Publication:** `publication/generated/` is produced only after exact replay and audit gates pass for every input. `paper/main.pdf` is a report of those retained artifacts.

## Fresh execution

Run from this directory with a new output path for every attempt:

```text
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 experiment.py develop --output runs/development-N
python3 experiment.py verify --input runs/development-N \
  --output runs/development-N/verification-exact.json --replay
python3 experiment.py freeze --input runs/development-N \
  --output runs/freeze-N.json
python3 experiment.py confirm --lock runs/freeze-N.json \
  --output runs/confirmation-N
python3 experiment.py verify --input runs/confirmation-N \
  --output runs/confirmation-N/verification-exact.json --replay
python3 audit.py runs/confirmation-N runs/confirmation-N/audit.json
python3 experiment.py analyze --input runs/confirmation-N \
  --lock runs/freeze-N.json --output runs/analysis-N.json
```

Real streams use the same frozen lock:

```text
python3 transfer.py run --dataset air --lock runs/freeze.json \
  --output runs/air-regression-N
python3 transfer.py verify --input runs/air-regression-N \
  --lock runs/freeze.json \
  --output runs/air-regression-N/verification-exact.json --replay
python3 audit.py runs/air-regression-N runs/air-regression-N/audit.json

python3 transfer.py run --dataset beijing --lock runs/freeze.json \
  --output runs/beijing-transfer-N
python3 transfer.py verify --input runs/beijing-transfer-N \
  --lock runs/freeze.json \
  --output runs/beijing-transfer-N/verification-exact.json --replay
python3 audit.py runs/beijing-transfer-N runs/beijing-transfer-N/audit.json
```

## Admission and non-overwrite rules

- Local model phases require at least 512 MiB reclaimable memory and a non-rising one-second macOS pageout probe.
- Process peak RSS must stay below 512 MiB; numerical state must stay at or below 128 KiB; each phase has a 1,800-second limit.
- Refused phases create retained admission receipts and no model attempts.
- Output directories and evidence files are create-only. Use a new run identity rather than overwriting.
- Failed cells remain in denominators. Never rerun only unfavorable seeds.
- Exact replay and the separate auditor are complementary checks under common project control, not independent reproduction.

## Determinism and identities

Every attempt records source/config/task identity, raw prediction trace, counters, final checkpoint, wall time, RSS, and terminal status. Gzip metadata is fixed. The freeze binds source hashes, data-manifest identity, protocol, selected configurations, and development artifact identity. The hosted notebook embeds a source/data archive and checks its SHA-256 before extraction.

## Publication regeneration

After valid confirmation and both transfer runs:

```text
MPLCONFIGDIR=/tmp/neurogenesis-v4-mpl /usr/local/bin/python3 \
  publication/generate.py \
  --lock runs/freeze.json \
  --analysis runs/analysis.json \
  --confirmation PATH/TO/confirmation \
  --air PATH/TO/air-regression \
  --beijing PATH/TO/beijing-transfer \
  --hosted-receipt hosted/hosted-verification.json \
  --output publication/generated
```

Copy or link the generated figures, tables, and `results.tex` into `paper/`, then compile `paper/main.tex`. The generator refuses missing, failed, non-exact, or unaudited evidence.

## Claim boundary

The project may report the exact registered synthetic result, descriptive outcomes on the two named chronological streams, and a completed verified Kaggle replay. It must not claim state of the art, general intelligence, production readiness, hardware-independent speed or energy gains, IID real-data confidence, cross-dataset generalization, peer review, or independent external replication.
