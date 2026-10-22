# Final research execution report

## Submission-readiness decision: **NOT READY FOR SUBMISSION**

This execution materially extends the repository and produces two new, verified successor studies, but it does not establish a publishable general learning claim.  The only positive result remains S1's narrow synthetic grouped-additive final matrix; S1 already failed its longer-horizon quality criterion.  The new structural-misspecification successor S2.1 also fails both registered endpoints.  The manuscript is therefore an evidence-bound draft describing a mechanism boundary and negative results, not a submission-ready paper.

## S3 hosted execution status: **REMOTE_SOURCE_UPDATE_REQUIRED**

The real-data GPU successor `DLML-S3-R0` remains frozen before protected-test execution. A clean-room audit proved that the historical private source dataset version 2 would fail the launcher's full preflight because required smoke configuration/protocol files were absent; the archive also contained macOS AppleDouble metadata. No protected target was opened. A corrected local artifact now passes the full suite and offline wheel build from its exact extraction, but it must be uploaded as a new private dataset version and reattached before submission. Kaggle authentication is unavailable in the connected browser profile, so that remote update is not claimed. See `HOSTED_EXECUTION_STATUS.md`.

This queued state is not a scientific result. S3 cannot alter the manuscript or any claim until its protected-test outputs, failures, hashes, manifest, and raw metrics have been independently checked.

## Implemented research system capabilities

- Extended the deterministic generator with controlled pairwise interactions, linear/quadratic/sinusoidal component bases, and learner-visible interleaved partitions that do not reveal the true generating partition or coefficients.
- Added a simultaneous (block-Jacobi) residual-target ablation, alongside the existing sequential/Gauss-Seidel analytic residual rule.
- Added optional affine-intercept capacity to every implemented learning condition, including checkpoint serialization and independent readback.  The default remains off, preserving S1's model definition.
- Added checkpoint-weight/intercept recording and verification of every saved checkpoint evaluation, not only final weights.
- Added `derived-learning analyze`, which produces deterministic paired bootstrap summaries and SVG learning-curve figures solely from finalized `results.jsonl` outputs.
- Added frozen S2 and S2.1 protocols/configurations, a safe one-command S2.1 reproduction script, a manuscript source/bibliography/generated tables, and a CPU/GPU status statement.
- Added a frozen real-data GPU successor using matched additive block MLPs, ridge and histogram-gradient-boosting baselines, leakage-resistant train/development/protected-test splits, complete resource accounting, and five predeclared seeds.
- Added a hash-verifying Kaggle launcher, fail-closed source builder, typed private/GPU/internet-off metadata, and independent hosted archive verifier. The corrected local source bundle SHA-256 is `fb24189f110a430f9868acc430aa487764cb6ab30a950dd00126b3bd139f5f86`.

## Executed, verified experiments

| Study | Scope | Final equal-resource NMSE: Adam / analytic | Long-horizon equal-resource NMSE: Adam / analytic | Registered result |
|---|---|---:|---:|---|
| S1 | Correctly partitioned additive linear tasks | 0.024140 / 0.003475 | 0.004701 / 0.004754 | Final pass; long-horizon quality fail |
| S2 | Wrong partition + nonlinear/interacting tasks, no intercept | 0.964820 / 0.958392 | 1.268040 / 1.263700 | Fail; retained but structurally confounded by mean offset |
| S2.1 | S2 successor with common fitted intercept | 0.724612 / 0.711624 | 0.822250 / 0.822343 | Fail on both predeclared criteria |

S2.1 used five paired final and five paired long-horizon task seeds, with all 50 final/long-horizon condition records successful.  Its final paired analytic-minus-Adam NMSE mean was -0.0129873 (descriptive deterministic bootstrap 95% interval [-0.0281100, 0.0021354]); its long-horizon mean was +0.0000932 ([-0.0003178, 0.0004638]).  These post-run intervals are descriptive and do not override the frozen composite endpoint.

The block-Jacobi ablation was retained despite numerical explosion under selected S2/S2.1 settings; this is evidence of a stability limitation, not a discarded cell.

## Verification receipts

- Unit suite: 30 tests passed after the final implementation, including S3 terminal-classification tests, Kaggle metadata/bundle contracts, and a complete generated hosted-evidence verification fixture.
- S2: manifest verification succeeded; independent terminal/checkpoint readback succeeded for 50 condition results.
- S2.1: manifest verification succeeded; independent terminal/checkpoint readback succeeded for 50 condition results.
- Package build: dependency-free wheel built successfully (`derived_learning_ml-0.1.0-py3-none-any.whl`).
- Manuscript: `paper/main.tex` compiled successfully with Tectonic; its rendered three-page PDF was visually inspected page by page.

## Artifacts

- S2.1 verified report and analysis: `runs/s21_structural_misspecification_intercept/REPORT.md`, `evaluation_verification.json`, `manifest_verification.json`, and `analysis/`.
- S2 retained negative run: `runs/s2_structural_misspecification_retry/`.
- Reproduction command: `sh scripts/run_s21_reproduction.sh NEW_OUTPUT_DIRECTORY` from the repository root.  It refuses to overwrite an existing output directory.
- Manuscript package: `paper/main.tex`, `paper/references.bib`, `paper/generated_results.tex`, `paper/figures/s21_learning_curves_final.svg`, and compiled `paper/main.pdf`.
- S3 hosted-execution receipt: private source dataset v2 and private notebook were configured, but clean-room qualification later invalidated v2 for launch. The corrected artifact is locally verified and awaits authenticated private dataset replacement. No GPU result is claimed.

## Skeptical-reviewer blockers

1. The method is mathematically a partial-residual/backfitting variant in its successful setting; no novelty claim beyond the controlled evidence/accounting package is justified.
2. S1's advantage does not persist on its registered long-horizon family.
3. S2.1 does not support the registered advantage under the tested wrong-partition, nonlinear/interacting structural shift.
4. The completed evaluation remains internally executed synthetic work; S3 supplies a frozen real-data admission track but has not executed, and there is no independent reproduction.
5. S3 implements nonlinear block models and an equal-example-pass matched control, but it is not a learned target controller and has no GPU outcome yet. Claiming a hosted result before artifact verification would be fabricated.
6. The five-seed matrices are useful falsification evidence but insufficient for broad statistical/generalization claims.
7. Wall-clock and peak-memory results require a distinct frozen cost protocol; current work units are deterministic algorithmic proxies.
8. The S2.1 paper draft does not yet include a rendered in-document vector figure; the underlying recorded SVG is packaged separately.  This is a packaging limitation, not missing experimental evidence.

## Checklist status

`RESEARCH_FEATURE_MEGA_CHECKLIST.md` now marks the completed protocol, task-generator, Jacobi-ablation, curve, and full-checkpoint-readback deliverables with direct evidence paths.  All other items remain unchecked; they are not silently converted into completed research work.

## Branch and workspace integrity

The dedicated branch `codex/derived-learning-execution` was created before implementation.  The shared checkout subsequently reports a different active branch, `codex/pi-ml-research-system`; no commit was made after that change so that unrelated parent-workspace deletions and files remain untouched.  All work above is present in the Derived-Learning-ML working directory and is uncommitted pending the user's preferred integration workflow.
