# NEUROGENESIS memory v3

Fully implemented standard-library Python reference study of bounded RLS predictor memory under delayed labels. This is the second and final mechanism version in the 2026-10-21 development budget.

Read the repository-root `RESEARCH_STATE.md`, then `SPECIFICATION.md`, `REVISION.md`, `protocol.json`, and the current run receipts before continuing. Do not retune this version using protected outcomes.

## Current outcome

Overall **MIXED**: all 1,260 frozen synthetic attempts completed and reproduced exactly; primary MSE is 36.01% lower with 58.55% fewer updates. Six of ten real-transfer methods failed numerically, preventing a candidate comparison. The implementation remains a research prototype with an unresolved robustness flaw. See root `RESULTS_CURRENT.md`. Kaggle execution and independent external replication are absent.

## What exists

- `engine.py`: frozen predictor bank, legal delayed-credit gating, NLMS/Adam/RLS, FIFO cache/replay, exact policy and runner checkpoint restoration, finite-value and capacity enforcement.
- `experiment.py`: complete registered development matrix, immutable attempts, selection freeze, protected confirmation, paired seed-level bootstrap analysis, and exact replay verification.
- `real_transfer.py`: hash-checked UCI data, hourly clock validation, missingness-preserving transformation, chronological warmup/test transfer, and verification.
- `audit.py`: separate arithmetic/timing/coverage auditor that does not import model or metric code.
- `test_engine.py`, `test_pipeline.py`: mathematical, leakage, control, capacity, missingness, serialization, and failure-retention tests.
- `publication/generate.py`: standard Matplotlib figures, CSV tables, numerical claim ledger, and LaTeX results generated from verified retained records.
- `paper/`: complete manuscript source and unchanged local refined `arxiv.sty`, with template provenance.

## Reproduce the implementation

Requires Python >=3.10; no third-party dependencies for models or experiments. Plotting requires Matplotlib. Tectonic builds the paper.

```sh
cd successors/memory_v3
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 experiment.py verify --input runs/development-01 --output /tmp/neurogenesis-dev-verify.json --replay
python3 experiment.py verify --input runs/confirmation-02 --output /tmp/neurogenesis-confirm-verify.json --replay
python3 publication/audit_arithmetic.py runs/confirmation-02 /tmp/neurogenesis-independent-audit.json
python3 publication/verify_real_outcome.py --input runs/real-transfer-03 --output /tmp/neurogenesis-real-evidence.json --replay
```

Use fresh output filenames; verification never overwrites retained receipts. Exact replays are verification, not new independent studies. Experiment commands are documented in the paper and `experiment.py --help`. The phase runner refuses output collisions, rising macOS pageouts, insufficient memory reserve, or process RSS above 512 MiB. Refused admission directories remain as history. Do not weaken these guards.

## Use a saved predictor

```python
import gzip, json
from engine import Policy, Config

path = 'runs/confirmation-02/attempts/base-90001__candidate/checkpoint.json.gz'
with gzip.open(path, 'rt') as file:
    checkpoint = json.load(file)
config = Config(**checkpoint['config'])
dimension = checkpoint['policy']['d'] - 1
policy = Policy.restore(checkpoint['policy'], config, dimension)
prediction, ticket = policy.predict([0.0] * dimension)
# Deliver a label only when legally available; ticket stores prediction-time outputs.
policy.observe([0.0] * dimension, 0.0, ticket)
```

For exact experiment continuation, use `Runner.restore(rows, config, checkpoint)` with the same task identity. A changed stream cannot be disguised as continuation of a frozen task.

## Scientific boundaries

The candidate combines established ideas. It is not a proven general-purpose neural model and is not compared against full CEP/DynaME implementations. A synthetic positive result does not imply real-data success. One real stream is descriptive transfer, not population-level replication. Weight updates, estimated coordinate arithmetic, wall time (including serialization), numerical state, and process RSS are reported separately. Independent external replication remains a separate requirement.

## Provenance

Version 1 and version 2 remain in their original paths. The v2 candidate failed against the strongest baseline on development. Version 3 changes both updater and gate; the main experiment does not isolate those two changes individually. The preconfirmation source commit and freeze digest are in `runs/preconfirmation_anchor.json`. The GitHub branch name identifies the overall memory campaign; the final model version is v3.

The original `real_transfer.py verify` intentionally rejects the failed real comparison. The supplementary auditor verifies preservation and exact reproduction of failures; it does not relax the efficacy gate.
