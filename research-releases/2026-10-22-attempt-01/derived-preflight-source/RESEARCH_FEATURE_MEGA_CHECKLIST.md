# Research feature mega-checklist

## Purpose and scientific north star

This is a delivery checklist for turning the current dependency-free **DLML-S1-P0** package into a research system, not for retroactively broadening S1.  S1's core contribution is a controlled comparison of exact analytic partial-residual targets against gradient-based updates on grouped additive linear regression, with split isolation and accounted work.  The correct scientific interpretation is that this is an instrumented backfitting/block-coordinate mechanism test.  It is **not** a demonstrated learned-target architecture or a general deep-learning result.

The most important finding to resolve is already recorded: the primary method passes the five-seed final family but loses its equal-resource quality advantage on the three-seed long-horizon family.  Every successor below must preserve S1 unchanged, use a new versioned protocol, and be allowed to produce a negative result.  A checked item must contain the stated artifact and acceptance evidence, not merely code.

## A. Freeze the reference and admit successors

- [ ] **A01 — Seal an S1 reference capsule.** Export the current source-tree/config/output hashes, 32 saved evaluation records, execution environment, and an immutable `S1_REFERENCE.json`; acceptance is a clean-machine verifier that reports every recorded hash and terminal NMSE as matched.
- [ ] **A02 — Write an S2 charter for structural misspecification.** State a pre-registered question about when residual targets help or fail with wrong groups and non-additive structure; acceptance is a signed charter that names the mechanism, admissible information, primary endpoint, and falsifier without changing S1 files.
- [x] **A03 — Freeze an S2 protocol before outcome generation.** Implemented as `PROTOCOL_S2.md` and the confound-corrected successor `PROTOCOL_S21.md`; each completed manifest records the declared protocol hash. Evidence: `runs/s2_structural_misspecification_retry/manifest.json` and `runs/s21_structural_misspecification_intercept/manifest.json`.
- [ ] **A04 — Build an experiment-schema validator.** Validate study IDs, mutually exclusive splits, candidate grids, budgets, hypotheses, and result paths before execution; acceptance is a failing test corpus for malformed/overlapping configurations plus successful validation of all shipped protocols.
- [ ] **A06 — Add a claim-admission gate.** Generate `CLAIM_STATUS.json` that only permits language supported by the registered endpoint and records failures/unknowns; acceptance is tests showing a long-horizon failure blocks a broad-superiority claim.

## B. Expand the task domain where the mechanism can break

- [x] **B01 — Generate controllable pairwise-interaction tasks.** Implemented deterministic within-true-block interaction terms with `interaction_strength`; exercised in S2/S2.1. Evidence: `src/derived_learning/tasks.py`, both frozen configs, and their saved `tasks.json` files.
- [x] **B02 — Generate nonlinear additive components.** Implemented linear, quadratic, and sinusoidal component bases while keeping coefficients/component values generator-internal. Evidence: `tests/test_tasks.py` and the quadratic S2/S2.1 task manifests.
- [x] **B03 — Generate wrong-partition tasks.** Implemented a learner-visible interleaved partition that remains a valid partition but mixes true components; task provenance records whether truth was exposed. Evidence: S2/S2.1 `tasks.json` and `test_structural_extensions_are_deterministic_and_do_not_expose_truth`.
- [ ] **B05 — Add ill-conditioned covariance families.** Produce Toeplitz, block-correlated, low-rank-plus-noise, and near-collinear designs with measured condition-number bins; acceptance is a diagnostic table confirming achieved covariance/condition ranges.
- [ ] **B06 — Add sample-starved and high-dimensional regimes.** Sweep `n/d` below, near, and above one with ridge-safe solvers; acceptance is no silent singular solve plus recorded conditioning and numerical-failure outcomes.
- [ ] **B07 — Add train/evaluation distribution shifts.** Independently vary covariance, coefficient scale, noise, sparsity, and feature means after fitting; acceptance is a held-out shift matrix with per-axis metadata and locked selection isolation.
- [ ] **B08 — Add heavy-tailed and contaminated noise.** Support Student-t, mixture outliers, and label contamination with declared rates; acceptance is an empirical contamination check and robust-loss baselines run on the same tasks.
- [ ] **B09 — Add missing-feature and missing-block simulation.** Mask support and/or evaluation features under MCAR and blockwise missingness; acceptance is explicit mask provenance, no evaluation-target access, and at least one imputation-aware comparator.
- [ ] **B10 — Add a real tabular regression admission track.** Define datasets, licenses, preprocessing, train/validation/test boundaries, and leakage audit for at least three suitable public datasets; acceptance is dataset cards plus reproducible raw-to-split manifests, not a claim based on a downloaded benchmark alone.

## C. Make the model family scientifically discriminative

- [ ] **C01 — Add a global closed-form ridge oracle.** Implement and account for the full-feature ridge solution as an attainable additive-linear ceiling; acceptance is an independent normal-equation test and comparison showing the residual rule's gap to the global optimum.
- [ ] **C02 — Add classical backfitting as a named baseline.** Implement equivalent cyclic partial-residual updates with transparent naming and identical solver settings; acceptance is a numerical-equivalence test that prevents relabeling classical backfitting as a novel method.
- [x] **C03 — Add block-Jacobi residual updates.** Implemented `BlockJacobiResidualRule` and ran it on all S2/S2.1 cells. It is retained as a numerically unstable ablation under selected S2/S2.1 settings, rather than omitted. Evidence: both generated reports and `selection.json` files.
- [ ] **C05 — Add adaptive regularization selected without final access.** Implement a development-selected or support-only, protocol-declared ridge policy; acceptance is a leakage test and a table separating fixed, tuned, and adaptive regularization costs.
- [ ] **C06 — Add convergence certificates for the linear setting.** Compute objective decrease, fixed-point residual, duality/KKT residual where applicable, and distance to global ridge; acceptance is per-checkpoint certificates that match an independently implemented calculation.
- [ ] **C07 — Add robust local solvers.** Provide Huber/iteratively reweighted local regression and compare it against squared-loss updates under B08; acceptance is retained convergence/failure records and a pre-specified robust-noise endpoint.
- [ ] **C09 — Add nonlinear block modules.** Introduce a small feature-map or shallow MLP module per block for B02 while retaining a declared local-target interface; acceptance is exact parameter/optimizer-state serialization and a comparison to an equally sized end-to-end model.
- [ ] **C10 — Add interaction-residual modules.** Allow selected cross-block pair modules to receive residual targets after main blocks; acceptance is a predeclared interaction budget, hidden true components, and ablations with interaction modules disabled.
- [ ] **C11 — Implement a learned target-controller successor.** Train a controller that proposes local targets or damping only from admissible support-side state; acceptance is a separate S3 protocol, controller-training work/calls in the resource ledger, and no reuse of S1's claim.
- [ ] **C12 — Add controller necessity and information ablations.** Compare learned controller, frozen/random controller, analytic target, no-target block gradient, and oracle-only-for-diagnostics conditions; acceptance is an ablation table that reports each controller's inputs and all training cost.
- [ ] **C13 — Add support-only trust-region/damping logic.** Accept or damp block updates using only support loss and numerically stable criteria; acceptance is a trace of proposed/accepted/rejected updates and a test proving evaluation labels cannot affect acceptance.
- [ ] **C14 — Add feature-group discovery as a separate module.** Infer candidate partitions from support features or supplied metadata, then evaluate sensitivity to its errors; acceptance is group-discovery provenance, fit/evaluation separation, and comparison with true-group diagnostic upper bound clearly labeled as oracle.

## D. Training execution, recovery, and fair comparisons

- [ ] **D01 — Generalize the training loop to minibatches.** Support deterministic shuffling, batch size, epoch, and update-frequency controls for every comparable method; acceptance is bitwise repeatability under fixed seeds and explicit sample-read accounting.
- [ ] **D02 — Implement resumable checkpoints for every condition.** Persist model, optimizer/rule state, RNG state, schedule state, config/protocol hashes, and resource counter atomically; acceptance is interrupted-versus-uninterrupted equality tests for Adam, analytic, and new rules.
- [ ] **D03 — Add failure-safe run state.** Use `RUNNING`, `SUCCEEDED`, `FAILED`, and `INTERRUPTED` states with retained exception type, last valid checkpoint, and traceback; acceptance is injected-crash tests that preserve partial evidence and never mark an incomplete run complete.
- [ ] **D05 — Implement resource accounting v2.** Record wall-clock, peak RSS, FLOP/operation proxy, sample reads, disk bytes, accelerator/device, controller training, and preprocessing separately; acceptance is an audited ledger with a unit test for every update rule.
- [ ] **D06 — Add wall-clock and memory benchmark harnesses.** Benchmark warm-up, repeated trials, thread/device policy, and peak memory on fixed hardware; acceptance is a versioned benchmark protocol with uncertainty intervals rather than single timing claims.
- [ ] **D07 — Add baseline families appropriate to each regime.** Run SGD/momentum, AdamW, L-BFGS or conjugate gradient, global ridge, coordinate descent/backfitting, and robust/sparse solvers when relevant; acceptance is a baseline registry that refuses an incompatible baseline/task pairing.
- [ ] **D08 — Standardize hyperparameter budgets across methods.** Allocate equal development task count, trials, budget, and selection criterion per tunable method; acceptance is a selection audit table listing every attempted candidate, failure, and total tuning work.
- [ ] **D09 — Add nested task-family selection.** Separate meta-train, development, final, and stress-test families for all extensions; acceptance is a programmatic information-flow test showing no final/stress target reaches configuration selection.
- [ ] **D10 — Add multi-seed execution plans.** Expand each admitted final matrix to a protocol-justified seed count with paired task identity across methods; acceptance is a run plan that detects omitted, duplicated, or unpaired cells before summary generation.

## E. Evaluation that can falsify the hypothesis

- [x] **E01 — Report learning curves at matched resource checkpoints.** Every S2/S2.1 checkpoint retains evaluation NMSE and work; the new `analyze` command emits saved-output SVG curves without interpolation or best-checkpoint selection. Evidence: `runs/s21_structural_misspecification_intercept/analysis/figures/`.
- [ ] **E02 — Add confidence intervals and paired tests.** Use predeclared paired bootstrap/permutation procedures for final endpoints and report effect sizes; acceptance is a deterministic analysis script that consumes saved runs only and preserves all seed-level values.
- [ ] **E03 — Correct for multi-regime comparisons.** Declare primary versus secondary endpoints and apply an appropriate multiplicity policy across S2/S3 regime families; acceptance is a report that labels exploratory analyses and cannot promote them to primary success.
- [ ] **E04 — Add time-to-threshold survival analysis.** Handle unreached quality thresholds as censored observations rather than assigning convenient values; acceptance is a generated table/curve with raw threshold events and censoring records.
- [ ] **E05 — Quantify sensitivity to budget.** Sweep short, equal, and long budgets so the observed S1 reversal is measured rather than hidden; acceptance is a predeclared crossover analysis showing where, if anywhere, advantage persists.
- [ ] **E06 — Quantify sensitivity to group error.** Plot effect versus partition corruption, missing groups, and discovered groups; acceptance is a generated phase diagram with confidence intervals and the exact task generator parameters.
- [ ] **E07 — Quantify sensitivity to non-additivity.** Plot effect versus interaction strength and nonlinear amplitude, including a regime that falsifies the analytic-target advantage; acceptance is an explicit failure boundary, not only favorable slices.
- [ ] **E09 — Add data-efficiency curves.** Vary support-example counts at fixed evaluation distributions and compare quality/work; acceptance is a plotted sample-complexity curve with identical tasks per method.
- [ ] **E10 — Add out-of-distribution robustness tables.** Summarize B07/B08/B09 metrics by shift mechanism, severity, and worst-case performance; acceptance is generated tables with no averaging across incompatible shifts.
- [ ] **E11 — Add numerical-stability stress tests.** Evaluate rank deficiency, extreme scale, near-collinearity, and solver tolerances; acceptance is retained failures plus condition numbers and stable/unstable classification.
- [x] **E12 — Add independent result readback for all checkpoints.** S2/S2.1 retain checkpoint weights (and intercepts for S2.1); `evaluate` reconstructs every stored checkpoint and terminal model. Evidence: both `evaluation_verification.json` receipts with 50 terminal results and successful checkpoint readback.

## F. Inspection tools that expose the mechanism

- [ ] **F01 — Produce residual-target decomposition plots.** For selected tasks, show label, each block prediction, partial residual target, and post-update residual over samples; acceptance is a plot generated solely from a saved trace and an underlying machine-readable table.
- [ ] **F02 — Visualize block-update trajectories.** Plot block weights, objective, fixed-point residual, and contribution norms by sweep; acceptance is a deterministic figure plus a test that trajectory points agree with checkpoints.
- [ ] **F05 — Build a partition-sensitivity heatmap.** Render quality and work across true/supplied group mismatch categories; acceptance is a reproducible heatmap with sample counts and confidence intervals in the exported data.
- [ ] **F06 — Add learned-controller introspection.** Log target corrections, damping, controller uncertainty, and input attribution for S3; acceptance is an exportable trace and an ablation confirming no evaluation information was consumed.
- [ ] **F07 — Add per-task comparison cards.** Create a self-contained HTML/Markdown artifact containing config, seed, curves, counters, final metrics, failures, and artifact hashes; acceptance is one card for every task-condition cell generated in CI.
- [ ] **F08 — Add error-slice inspection.** Stratify prediction error by covariance/condition bins, noise/outlier status, sparsity, and shift severity; acceptance is a saved prediction schema and generated slice table with minimum-count suppression rules.

## G. Experiment interface and artifact operations

- [ ] **G01 — Expand the CLI into an experiment lifecycle interface.** Provide `validate`, `plan`, `run`, `resume`, `status`, `compare`, `plot`, `verify`, and `export` commands with stable exit codes; acceptance is CLI integration tests and `--help` examples that execute against a smoke study.
- [ ] **G02 — Add a dry-run resource planner.** Estimate cells, checkpoints, arithmetic proxy, disk, and expected task count from a config before execution; acceptance is the plan embedded unchanged in the run manifest and checked against completed cells.
- [ ] **G03 — Add a run registry.** Maintain append-only run IDs, statuses, parent protocol, source/config hashes, and artifact locations; acceptance is a query that detects duplicate run IDs and conflicting provenance.
- [ ] **G04 — Add comparison commands with compatibility checks.** Refuse to pool runs with different protocol versions, task hashes, or accounting versions unless explicitly requested as a labeled cross-protocol comparison; acceptance is negative tests for each incompatibility.
- [ ] **G05 — Add a local read-only dashboard.** Serve run status, queued/resumed cells, curves, resource ledgers, comparison tables, and claim status without modifying artifacts; acceptance is a browser smoke test over a saved run and exported static report.
- [ ] **G07 — Add artifact export bundles.** Export results, figures, tables, configs, protocols, environment lock, and checksums into a named release bundle; acceptance is extraction into a clean directory followed by successful verification.
- [ ] **G08 — Add a redacted public-data mode.** Allow public demonstration artifacts to omit sealed evaluation labels, hidden coefficients, and local paths while retaining verifiable hashes; acceptance is a scanner test proving those fields are absent.

## H. Reproducibility, paper evidence, and public demonstration

- [ ] **H01 — Create one-command successor reproductions.** Provide commands that create an isolated output directory, run an admitted protocol, verify every artifact, and generate the report; acceptance is a clean-environment receipt with source/config hashes.
- [ ] **H02 — Add deterministic-environment capture.** Record Python/OS/CPU, dependency lock or dependency-free assertion, BLAS/thread variables, random seeds, and locale; acceptance is a verifier warning when a rerun's environment differs.
- [ ] **H04 — Build an external-reproduction packet.** Include source archive, protocol, commands, expected hashes/metrics, data acquisition instructions, and a result-submission schema; acceptance is a separate operator completing the packet or a clearly retained `NOT_YET_INDEPENDENTLY_REPRODUCED` status.
- [ ] **H05 — Generate paper-ready main figures from recorded evidence.** Produce learning-curve, budget-sensitivity, partition/non-additivity phase, and resource-tradeoff figures from saved outputs only; acceptance is figure metadata that names the run IDs and source-data hashes.
- [ ] **H06 — Generate publication tables from the registry.** Produce final/robustness/ablation tables with seed counts, uncertainty, work definition, failed cells, and claim status; acceptance is a test that no numeric table cell is hand-entered.
- [ ] **H07 — Produce an ablation matrix with mechanism labels.** Distinguish analytic partial residual, schedule, local solver, controller, group source, and oracle diagnostics; acceptance is a table that prevents conflating a learned controller with the analytic S1 method.
- [ ] **H08 — Write a limitations artifact tied to results.** Generate a section that cites the observed long-horizon reversal, synthetic-only scope, partition assumption, and uncompleted independent reproduction; acceptance is a report test that requires these limitations whenever applicable evidence exists.
- [ ] **H09 — Provide an interactive mechanism demo.** Build a local web demo/notebook where users vary correlation, noise, groups, interaction strength, and budget, then see matched curves and residual decomposition; acceptance is a smoke test and a visible in-app warning that it is an illustrative synthetic experiment, not an efficacy claim.
- [ ] **H10 — Provide a scriptable inference and trace demo.** Offer an API/CLI that accepts a declared grouped tabular support set, performs fitted updates, returns predictions plus a provenance-safe trace, and refuses undeclared/malformed partitions; acceptance is an end-to-end example with saved input/output hashes and no hidden-label requirement.

## Completion criteria

This checklist is complete only when its checked items are backed by recorded artifacts and the successor studies have outcomes—positive, negative, or inconclusive—under frozen protocols.  Passing code, a dashboard, or a polished manuscript alone does not establish scientific validity, generalization, publication readiness, or independent reproduction.
