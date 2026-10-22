# Execution log

## 2026-10-19 — intake and data qualification

- Opened folder was empty; parent Atlas checkout state preserved.
- Selected the explicitly authorized fallback property task: Matbench v0.1 experimental band gap from composition.
- Pinned dataset SHA-256 `783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2`.
- Pinned official validation SHA-256 `94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d`.
- Pinned extracted task split SHA-256 `e36a0b8c1688fdd8623f411964eb3d5c9759554cef3ae575aa88b99fb84f5979`.
- Pinned Matminer metadata SHA-256 `a637a1fe146b70f5f663fd480a2475362dc446741893c3048ec992219ba2a0b0`.
- Validated 4,604 rows and exact official identifiers.
- Found 3 canonical duplicate groups / 6 rows / 4,601 unique compositions.
- Froze protocol before model execution.
- Pre-experiment suite: 29 passed in 5.46 s; post-repair suite: 29 passed in 2.10 s.

## `development-v1`

- Exact command: `/usr/bin/time -l .venv/bin/python -m ml4materials run --run-id development-v1`
- Outcome: `FAILED_RETAINED`
- Refusal: pageouts 522,494 → 522,496 at `data_validation`.
- No model checkpoint completed.

## Engineering repair

- Preserved `development-v1` unchanged except receipt enrichment and refreshed artifact hash manifest.
- Reused the already extracted and SHA-256-bound 585 KiB task split instead of reparsing the 44 MiB all-task bundle during every run.
- Assigned writable temporary Matplotlib/font caches.
- Did not change the data, folds, features, models, seeds, endpoints, 512 MiB limit, or no-pageout condition.
- Verification suite: 29 passed.

## `development-v1-r2`

- Exact command: `/usr/bin/time -l .venv/bin/python -m ml4materials run --run-id development-v1-r2`
- Outcome: `FAILED_RETAINED`
- Completed engineering checkpoints: official folds 0 and 1, each serialized and required to reload identically before progress.
- Refusal: pageouts 522,743 → 522,782 after `official_fold_1`.
- No aggregate endpoint was calculated or claimed.

## Final software verification

- Expanded suite initially found two incorrect fixture expectations; the failed JUnit XML was retained and only the assertions were corrected.
- Final suite: 39 passed in 5.19 s; 67% statement coverage.
- Python bytecode compilation: pass.
- Offline lock consistency: pass; 29 resolved packages.
- Source distribution and wheel: built successfully.
- Wheel installation in a clean Python 3.12 environment: pass.
- Installed-wheel checkpoint prediction/uncertainty smoke test: pass using the partial fold-0 engineering checkpoint only; outputs are not scientific results.
- Both failed-run artifact manifests re-generated and verified without missing, changed, or unmanifested files.

## 2026-10-20 — research execution branch

- Created branch `codex/ml4materials-research-execution`; pre-existing parent-checkout changes remain untouched.
- Fixed a reproducibility blocker in plotting: sandboxed macOS font inventory returned malformed `system_profiler` data. Static plots now use Matplotlib's bundled fonts through `MPL_IGNORE_SYSTEM_FONTS=1`; the Agg backend remains headless and deterministic in this environment.
- Added configuration planning, resource preflight with durable receipt, run registry, and fail-closed completed-run comparison commands. The v1 preflight was admitted at 122.7 MiB RSS with zero pageout delta at that instant.
- `development-v1-r4` was attempted with the unmodified frozen protocol. Fold 0 serialized and reload-validated; the guard refused after pageouts rose from 1,018,149 to 1,018,190. Receipt and manifest are retained.
- Implemented direct-quantile gradient-boosting interval models, interval scoring, versioned quantile checkpoints, a separate successor protocol/configuration, and `run-quantile`.
- `quantile-composition-v2-r1` was attempted under the same resource contract. Fold 0 serialized and reload-validated; the guard refused after pageouts rose from 1,023,792 to 1,023,941. No v2 metric is reported.
- Added a LaTeX research-execution record, bibliography, and a submission release gate. The LaTeX source compiled successfully; the release gate correctly rejected the retained failed run.
- A fresh `development-v1-r3` attempt passed instant preflight but was refused at data validation after pageouts rose from 1,036,491 to 1,036,530; the immutable failure receipt and manifest are retained.
- Repaired the completed-run status contract (`SUCCESS` versus the release gate's `COMPLETED`) and added a positive-path release-gate test.
- Extended the unchanged no-pageout condition to Linux by monitoring `/proc/vmstat` `pswpout`, and added `scripts/run_kaggle.sh` for test, preflight, frozen execution, verification, release gating, registry generation, and hashed result packaging.
- Verification after these repairs: 49 tests passed.


## 2026-10-21 — bounded robust-aggregation successor and closeout

- Preserved historical v1/v2 protocols, outputs and failed runs; archived stale summaries.
- Created `RESEARCH_STATE.json`, a 30-item flaw audit and a primary-source literature boundary.
- Froze one grouped robust-aggregation development configuration before outcomes.
- Implemented complete training, inference, model/baseline/ablation/probe/holdout matrix and checkpoint checks.
- Hosted v3: 58 tests passed; synthetic seven-cell end-to-end check passed.
- Completed 15 grouped outer cells and five element holdouts on Kaggle: 424.255 s, peak RSS 372.852 MiB, zero pageouts.
- Median MAE 0.425622 versus same-tree mean 0.473870 eV; every repetition exceeded the 0.01 eV threshold. Primary disposition SUPPORTED.
- Retained adverse results: RMSE and positive-gap error worsened; Ga/Se/Te MAE worsened and OOD coverage remained poor. Overall scope MIXED.
- Independent standard-library verifier reproduced historical and new row-level calculations, group membership, ranking and resource checks.
- Source upload review initially required ownership evidence; authenticated --mine listing established the destination. Full extracted bundle scan found no secrets.
- Refreshed the expired existing Kaggle OAuth session through the official SDK; no credentials were printed.
- Recovered the fuller archived arXiv style with byte-level provenance; retained the earlier unrelated-header shim.
- Full manuscript source and data-generated paper assets prepared; release verification is recorded separately.

- Release-only engineering fix: deferred plotting imports until actual experiment/report execution. This avoids font inventory and Matplotlib startup during checkpoint inference; no training, split, calibration, or aggregation calculation changed. Frozen hosted source inventories remain unchanged.

### Final release verification, 2026-10-21
All 11 paper pages visually inspected. Corrected an equation unit-formatting typo to 0.05 eV, matching the executed configuration, and re-rendered page 3. Final source tests: 58 passed. Offline wheel/sdist build and clean installed-wheel inference passed; lazy plotting imports preserve exact saved predictions. Paper source archive and frozen Kaggle source transport retained in releases/20261021. No independent retraining or arXiv submission is claimed.

### GitHub publication, 2026-10-21
Published payload commit 0083fe78e7e397f0aa38f47af84843c24df32795 on codex/ml4materials-evidence-release-20261021 in private build-the-future-11/Atlas. Remote branch SHA and project tree 6908817fbcd5b8ddfac6e669da7a143e6b7c814d independently matched. Draft PR: https://github.com/build-the-future-11/Atlas/pull/54. All 276 payload files matched workspace bytes, secret scan passed, and only the project subdirectory was committed. Main is not merged; no arXiv submission or independent scientific replication is claimed.

### Author metadata supplied, 2026-10-21
User confirmed the paper author as Ryan Gomez with no affiliation supplied. Updated the title page and PDF author metadata, regenerated the paper and arXiv source package, and reconciled the current author-completion checklist. Scientific results and frozen run artifacts are unchanged.

### Adversarial release hardening and exact reproduction, 2026-10-21
A repeated user request triggered a bounded verification follow-up, with zero new scientific configurations. Fixed report regeneration that reset published state and authorship requirements; prohibited overwrite of the original freeze and transport. Strengthened numerical matrix/summary/source/subgroup/ranking verification. Four independently rehashed corrupted evidence copies passed the old verifier and were rejected by the repaired verifier; original run bytes were unchanged. Added independent arithmetic reproduction of the fixed bootstrap and random-ranking streams. Release tests: 72 passed. Submitted one fresh Kaggle execution of the exact original frozen bundle with a new run ID under the unchanged guards.

The fresh reproduction completed in 369.760 seconds at 372.066 MiB peak RSS, zero pageout increase. Predictions, units, splits, all 20 checkpoints, diagnostics, and scientific summaries (excluding timing) match exactly. The executed source identity and recorded numerical dependencies match. This establishes same-operator computational reproducibility only. The repaired report generator was executed and preserved published state and Ryan Gomez authorship.

### Local post-release verification and manuscript hardening, 2026-10-21

- Re-ran the full release suite: 72 passed.
- Independently re-verified both retained v3 runs from raw rows and manifests; both passed.
- Recomputed the 5,000-resample interval and 12 random-ranking cells for both runs; exact reproduction remained confirmed, with all 20 checkpoint hashes equal.
- Expanded the primary-source literature boundary with MODNet, task-specific SAASBO-CrabNet, and MD-HIT redundancy-control evidence. This adds flaw F38 and strengthens the no-SOTA/generalization boundary; no benchmark or scientific disposition changed.
- Repaired a collapsed connector in the generated workflow figure, regenerated all evidence-derived paper assets, compiled the manuscript, and visually inspected all 12 final pages.
- Final local paper SHA-256: `0c21f22650a59e7b710b6704bd0a8dfc1b1a2ef55abc754c979bf38303dfc11d`.
- The existing private GitHub payload, draft PR, release 0.1.1, and arXiv source package were not changed or resubmitted. These local post-release edits remain unpublished.

## 2026-10-22 — Kaggle readiness verification

- Fetched and compared the private `codex/ml4materials-evidence-release-20261021` remote ref (`fd67abdf30d21e6ed3ccfc92fd8df37cac360e81`); before this receipt, it matched the complete local project tree byte-for-byte.
- Re-ran the release suite: 72 passed. Independently recomputed the primary v3 row-level evidence, exact original-versus-reproduction comparison, and fixed bootstrap/ranking resampling checks; all passed.
- The existing Kaggle launchers and retained frozen bundles are ready for an immutable v3 reproduction or verification execution. No new scientific configuration or hosted run was submitted: the v3 development budget is complete, and score-seeking reruns are prohibited.
- Published the complete verified local closeout and this readiness receipt to the existing private release branch. The remote ref was confirmed at `2903f94b22467476051ea546e0778672c282368b` before this publication-state correction; no arXiv submission or new Kaggle run occurred.
- Added and tested `scripts/stage_kaggle_replay.py`, which creates a private Kaggle replay submission with a validated new run ID from the retained source bundle without modifying the historical launcher. The staging tests passed (5), and the full suite passed (77). No hosted execution was submitted.
