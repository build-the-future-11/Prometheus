# Results

Status: **NOT_EVALUATED_RESOURCE_BLOCKED**

No complete scientific result exists. The frozen development experiment was attempted four times and the no-rising-pageout admission rule refused every run. The second attempt completed and serialized two of five official fold models; another attempt serialized fold 0; the latest attempt stopped at data validation. Partial folds are retained as engineering evidence only and are not averaged, extrapolated, or presented as prediction performance.

## Completed work

- Established the previously empty project as one narrow composition-only experimental band-gap study; no additional property project was invented.
- Downloaded and hash-bound Matbench v0.1 data, full official validation bundle, exact task split manifest, and Matminer metadata.
- Parsed and validated all 4,604 formulas and targets.
- Preserved exact official five-fold IDs and separately defined group-isolated calibration and element-holdout contracts.
- Found and retained three upstream canonical duplicate groups (six rows) with target spans 0.58, 0.64, and 2.30 eV.
- Found canonical-composition train/test overlap in official folds 0–3; official results, if later completed, must carry this caveat.
- Implemented train-median, ridge, three-seed ExtraTrees, uncertainty scaling, split-conformal intervals, abstention, subgroup analysis, unique-composition candidate ranking, five element holdouts, label permutation, checkpoint reload checks, figures, ledgers, and report generation.
- Implemented CLI interfaces for preparation, training/evaluation, checkpoint prediction with uncertainty, label-blind CSV ranking, and run-manifest verification.
- Pinned the full environment in `uv.lock` on Python 3.12.
- Passed 49 software and data-integrity tests; the numerical orchestrator remains partly uncovered because the guarded full run did not complete.

## Exact execution receipts

### `development-v1`

- Outcome: `FAILED_RETAINED`
- Scientific cells/folds completed: 0
- Refusal phase: data validation
- Pageouts: 522,494 → 522,496 (delta +2)
- Command time: 24.57 s real, 19.55 s user, 1.31 s system
- Failure evidence: `runs/development-v1/failures.json`
- Receipt: `runs/development-v1/receipt.json`

### `development-v1-r2`

- Engineering repair before attempt: reuse the 585 KiB hash-bound task split instead of reparsing the 44 MiB all-task bundle; cache paths made writable. The pageout gate was unchanged.
- Outcome: `FAILED_RETAINED`
- Complete scientific study: no
- Checkpoints serialized and reload-gated: official folds 0 and 1 only
- Refusal phase: after official fold 1
- Pageouts: 522,743 → 522,782 (delta +39)
- Command time: 31.44 s real, 29.80 s user, 1.10 s system
- Failure evidence: `runs/development-v1-r2/failures.json`
- Receipt: `runs/development-v1-r2/receipt.json`

### `development-v1-r4`

- Outcome: `FAILED_RETAINED`
- Complete scientific study: no
- Engineering checkpoint: official fold 0 only, serialized under the unchanged frozen model configuration.
- Refusal phase: after official fold 0
- Peak process RSS: 163.9 MiB (below the 512 MiB ceiling)
- Pageouts: 1,018,149 → 1,018,190 (delta +41)
- Failure evidence: `runs/development-v1-r4/failures.json`
- Receipt: `runs/development-v1-r4/receipt.json`

### `development-v1-r3`

- Outcome: `FAILED_RETAINED`
- Complete scientific study: no
- Scientific cells/folds completed: 0
- Refusal phase: data validation
- Peak process RSS: 136.3 MiB (below the 512 MiB ceiling)
- Pageouts: 1,036,491 → 1,036,530 (delta +39)
- Failure evidence: `runs/development-v1-r3/failures.json`
- Receipt: `runs/development-v1-r3/receipt.json`

### `quantile-composition-v2-r1` successor

- Outcome: `FAILED_RETAINED`
- Scope: separate direct-quantile gradient-boosting interval study; it does not alter frozen `development-v1`.
- Engineering checkpoint: official fold 0 only, with reload-validated point/interval checkpoint serialization.
- Refusal phase: after official fold 0
- Peak process RSS: 160.3 MiB (below the 512 MiB ceiling)
- Pageouts: 1,023,792 → 1,023,941 (delta +149)
- No successor prediction, interval, or baseline metric is reported.
- Failure evidence: `runs/quantile-composition-v2-r1/failures.json`
- Receipt: `runs/quantile-composition-v2-r1/receipt.json`

The retained fold checkpoints are not a benchmark receipt. No official out-of-fold MAE, calibration curve, abstention result, ranking utility, element-holdout result, candidate ledger, or results-derived numerical manuscript was completed.

## Negative findings

- The upstream artifact contradicts its own metadata-level duplicate claim for three canonical composition pairs; every pair exceeds the stated 0.1 eV span rule.
- The official Matbench folds are not fully isolated under local scale/order-invariant composition canonicalization.
- The approved 512 MiB/no-rising-pageout environment did not admit a five-fold study with the frozen 288-tree ensemble, even after removing redundant full-bundle parsing.
- There is no evidence that the proposed model outperforms ridge, calibrates well, improves ranking utility, generalizes to unseen elements, or identifies a stable/useful material because those endpoints were not fully executed.

## Incomplete work and exact next commands

Do not retry on the current macOS host; four attempts have established that it does not provide a stable no-pageout window. Use the versioned Kaggle runner, which monitors Linux `pswpout`, and do not reduce trees, change seeds, weaken the gate, or reuse an existing run ID.

P01 — official prediction, calibration, abstention, subgroup and permutation endpoints:

```bash
scripts/run_kaggle.sh kaggle-development-v1-r1
```

P01-R — fixed-pool candidate ranking and ledger: produced automatically by the same successful `development-v1-r3` command after all official folds; no separate shortcut is valid.

P01-G — five element-holdout studies: produced automatically after ranking by the same successful command; no partial-fold substitute is valid.

After and only after a successful run:

```bash
.venv/bin/python -m ml4materials verify --run-id kaggle-development-v1-r1
.venv/bin/python -m pytest -q
python scripts/reproduce_submission.py --run-id kaggle-development-v1-r1
```

## Claim boundary

Software-complete and fixture-tested is not numerically executed science. Partial checkpoints are not official benchmark results. A future ranked composition would still not be independently verified stability, synthesis, safety, novelty, or utility.

