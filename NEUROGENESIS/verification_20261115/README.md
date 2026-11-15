# NEUROGENESIS v4 transfer evidence reconstruction

Date: 15 November 2026.

## Completed result

Both previously reported real-stream matrices have been reconstructed from the unchanged frozen source, data and selection lock. All **22 method attempts completed**, with **zero failed methods**. Exact replay validates **295,647 prediction-trace rows**. Separate arithmetic audits, full-lock reconciliation and an independent internal review pass.

This is frozen-code reproduction and evidence reconstruction. The October 21 manuscript already disclosed the Air and Beijing outcomes. These runs do not supply a fresh protected confirmation. The original raw transfer runs were not recovered, so agreement with the paper does not certify when or where the historical execution occurred. The frozen producer retains its original `protected_transfer` label for Beijing; the new execution receipts explicitly record the retrospective scope.

## Reconstructed results

MSE is expressed in each dataset's original target units squared. MSE values should be compared within a dataset, not across datasets.

| Stream | Candidate MSE | Selected RLS MSE | Candidate MSE change | Candidate / RLS solves | Solve reduction | Last-observation MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Air Quality | 1.211376300 | 0.694981527 | +74.303381% | 341 / 4,668 | 92.694944% | 3.228986718 |
| Beijing PM2.5 | 4809.104897076 | 5581.184054864 | -13.833609% | 3,002 / 8,661 | 65.338875% | 2494.100681215 |

Air remains negative: the gate sacrifices too much predictive accuracy despite suppressing most solves. Beijing satisfies the original descriptive candidate-versus-selected-RLS rule (MSE at most 1.02 times baseline and at least 20% fewer solves). Last-observation persistence has lower error than every learned method on Beijing; candidate MSE is 1.928192 times persistence MSE. The always-update bank also has lower Beijing error than the gated candidate. These results support a bounded mixed comparison, not a general forecasting advantage.

All three headline percentages in the old manuscript match to the originally reported two decimal places: Air +74.30% MSE, Beijing -13.83% MSE, and Beijing 65.34% fewer solves. `RESULTS.json` records the full precision comparison; `method_results.csv` contains all 22 method rows.

## What changed

The additions reconstruct missing raw evidence and close reproducibility checks around the frozen study. The predictor, data transforms, source files, protocol, selection lock, thresholds and historical ledgers remain byte-preserved.

The additive `reconcile_transfer_evidence.py` validator binds the method roster to the original lock, independently recomputes evaluation MSE using `math.fsum`, checks units and all decision fields, validates resource admission, recomputes the separate auditor's artifact digest, and binds every current run file and command log to the preserved successful execution receipt. It reports observed trace counts even for partial failed attempts.

The old transfer verifier accepted six corrupted evidence variants. The supplemental validator rejects a false support flag, a deleted comparison method, incorrect units, denied resource admission, a stale arithmetic digest, and an altered exact-replay receipt. An untouched copied bundle remains accepted. `verifier_mutation_receipt-v2.json` retains all seven checks. An independent internal review of the final validator and both real runs also passed; this is not independent investigator reproduction.

## Provenance and resources

- Source: `build-the-future-11/Prometheus`, commit `3323b449126ab487bc34756eac781b244ddef22b`.
- Frozen subtree: `NEUROGENESIS/successors/memory_v4`; all 11 source-map hashes match the original `runs/freeze.json`.
- Data: the UCI Air Quality and Beijing PM2.5 files match their frozen SHA-256 identities in `data/manifest.json`.
- Runtime: CPython 3.12.14 on Linux x86-64; standard-library model and verification code.
- Air: 9,357 chronological rows; 3,742 warmup rows and 5,615 evaluation rows, with 4,668 observed evaluation targets.
- Beijing: 17,520 rows for 2013–2014; 8,760 warmup rows and 8,760 evaluation rows, with 8,661 observed evaluation targets.
- Largest producer peak RSS: 111,902,720 bytes, below the 536,870,912-byte cap. Largest numeric state: 6,176 bytes, below the 131,072-byte cap. Both producer commands remain below the original 1,800-second phase bound.
- All 26 existing frozen implementation tests passed. Exact replay and the separately implemented arithmetic/timing auditor passed for both streams.

The Air predictive result is not a gate admitting Beijing. The frozen contract selects both streams independently; the numerical, resource, source and lock gates remain binding. All Air methods completed and all evidence checks passed before the already reported Beijing stream was reconstructed. `beijing_progression_basis.json` records this decision.

## Evidence layout

Complete raw archive: `NEUROGENESIS_Transfer_Reconstruction_2026-11-15.zip` (29,420,870 bytes). SHA-256: `6135d280bbde024b075518bd46c69687d6f5e0671a7b53f2c7785e06279214bd`. `ARCHIVE_RECEIPT.json` records its exact identity and the successful post-packaging payload checks. The compact Ryan branch carries the reproducible runner, validator, numeric summaries, command logs and raw-artifact digests. The full raw archive is delivered separately to the owner.

The complete scientific archive contains 49 files for each reconstructed stream: admission and task manifests; 11 sets of attempt metadata, raw traces, checkpoints and results; summary; exact replay; and separate arithmetic audit. It also contains the exact frozen source/data/lock needed to verify those outputs and the command logs/receipts in this directory. The original synthetic confirmation artifacts are outside this reconstruction's scope.

- `air-reconstruction/` and `beijing-reconstruction/`: commands, logs, whole-run digests and the final `reconciliation-v2.json` receipts.
- `RESULTS.json` and `method_results.csv`: complete numerical findings and limitations.
- `independent_review_final.json`: final internal review of source, arithmetic, gates and integrity checks.
- `source_freshness_receipt.json` and `prior_manuscript.json`: recovery scope and historical outcome exposure.
- `prospective_transfer_execution.json`: a preserved historical planning record. Its pending freshness review and initial run IDs were superseded by the discovery of already reported outcomes, the progression record and the reconstruction execution receipts. It is not the current status.
- `verifier_fixture_failure*.json`: two repaired metadata-test fixture problems (cross-filesystem hardlinks and formatting the positive control). Neither executed nor changed model outcomes.

## Verify a retained reconstruction

From the root of the complete archive, choose a new output receipt path:

```bash
python3 NEUROGENESIS/verification_20261115/reconcile_transfer_evidence.py \
  --source NEUROGENESIS/successors/memory_v4 \
  --run NEUROGENESIS/successors/memory_v4/runs/air-regression-reconstruction-20261115-01 \
  --execution-receipt NEUROGENESIS/verification_20261115/air-reconstruction/execution.json \
  --output /tmp/neurogenesis-air-check-new.json
```

Use the corresponding Beijing paths for the other stream. This command verifies retained evidence without generating new model outcomes. `execute_frozen_transfer.py` provides an explicit separate reproduction runner that requires a new run ID and receipt directory, preserves failures and refuses to overwrite existing outputs. Reproductions must keep the declared source/data/lock and resource limits.

## Remaining limitations

The work closes a missing local raw-evidence gap and repairs additive validation. It does not establish Kaggle execution, external investigator reproduction, cross-dataset generalization, universal improvement, production readiness or venue acceptance. The old frozen hosted-verification wrapper has a separately identified missing-digest issue; these local receipts do not claim to have executed or repaired a hosted campaign. No venue submission or main-branch merge occurred in this reconstruction.
