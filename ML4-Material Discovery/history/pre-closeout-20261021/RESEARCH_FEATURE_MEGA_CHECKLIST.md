# Research feature mega-checklist

## Scientific direction and completion rules

This project’s present contribution is an evidence-bound workflow for predicting **experimental inorganic band gap from composition only**, attaching split-conformal uncertainty, and retrospectively ranking a fixed historical pool for a 1.1–1.6 eV target window. It has reliable data contracts, formula invariance, official Matbench split handling, candidate provenance, and retained partial execution evidence. It does **not** yet have a completed five-fold numerical result: `development-v1` and `development-v1-r2` remain `NOT_EVALUATED_RESOURCE_BLOCKED` because the frozen no-rising-pageout guard stopped them.

The scientific system should therefore grow through explicitly versioned successor protocols. Do not overwrite failed runs, change `development-v1` in place, or present engineering checkpoints as benchmark results. Every checkbox below names a concrete deliverable and the evidence required to mark it complete. A passing test, a rendered fixture figure, or a dashboard implementation is not a scientific result unless it is bound to a successful recorded run.

## Executed evidence — 20 October 2026

- `development-v1-r4` and `quantile-composition-v2-r1` both retained fold-0 engineering checkpoints but were refused by the unchanged no-rising-pageout guard. Neither has reportable model metrics.
- Direct-quantile interval models, proper interval scoring, versioned quantile checkpoints, the `run-quantile` command, and fixture tests are implemented. `configs/studies/quantile-composition-v2.json` and `PROTOCOLS/quantile-composition-v2.md` bind this successor to the same pinned data while preserving v1.
- `artifacts/preflight-20261020.json` records an admitted preflight; `artifacts/run_registry-20261020.json` indexes retained manifests. `paper/main.tex` compiled successfully and the release gate refuses all current incomplete runs.

## A. Core model capabilities

- [ ] **A01 — Complete the frozen `development-v1` study in an admitted environment.** Deliver `development-v1-r3` (or later unused ID) with all five folds, ranking, holdouts, permutation control, resource receipt, and a verified artifact manifest; accept only if the unchanged 512 MiB/no-pageout gate passes.
- [ ] **A02 — Create a successor study registry.** Deliver `configs/studies/<study-id>.json`, immutable protocol text, rationale, preregistered endpoints, and relationship to the frozen predecessor for every changed hypothesis; accept only if no frozen v1 file is mutated.
- [ ] **A03 — Add composition-graph representations.** Deliver a composition graph encoder using element nodes and stoichiometric edges or a reproducible equivalent, with train-only fitted state and checkpoint serialization; accept with formula order/scale invariance tests and out-of-fold predictions.
- [ ] **A04 — Add physically motivated elemental descriptors.** Deliver a versioned descriptor set covering electronegativity, covalent/atomic radius, valence-electron proxies, periodic-row/group statistics, and mixing contrasts; accept with source/version/license manifest and a leakage audit proving train-only transforms.
- [ ] **A05 — Add a stoichiometry-aware attention model.** Deliver a Set Transformer or attention pooling model that returns element-level weights for each formula; accept with deterministic inference, saved checkpoints, and tests that identical canonical formulas yield identical predictions and attention after canonical ordering.
- [ ] **A07 — Add a crystal-structure model track.** Deliver a separately scoped structure-required study using a pinned, licensed structure source and graph network baseline; accept only after composition/structure identity joins, polymorph handling rules, and structure-availability selection bias are recorded.
- [ ] **A08 — Build a composition-plus-structure fusion model.** Deliver a gated model that can use composition alone or composition plus structure, reporting which modality was available; accept with missing-modality tests and matched-subset comparisons rather than pooled claims.
- [ ] **A09 — Add a reference first-principles signal study.** Deliver a clearly labeled successor that tests whether a pre-measurement calculated-gap feature helps on a provenance-safe subset; accept only with temporal/availability rules and results separated from the composition-only claim.
- [ ] **A10 — Build a multi-task property model.** Deliver a model jointly predicting experimental gap and only properly licensed, predeclared auxiliary properties; accept with per-task missing-label handling, a negative-transfer analysis, and single-task matched controls.
- [ ] **A11 — Add quantile regression intervals.** Deliver quantile models for 50/80/90/95% prediction intervals alongside normalized conformal intervals; accept with out-of-fold coverage, width, sharpness, and conditional-coverage tables.
- [ ] **A12 — Add ensemble diversity control.** Deliver bootstrap, feature-subsampling, seed, and architecture diversity variants that expose member correlations; accept with an ensemble-size curve and retained individual-member predictions.
- [ ] **A13 — Add distributional regression.** Deliver a heteroscedastic likelihood model producing a full predictive distribution; accept with proper scoring rules (NLL/CRPS), calibration diagnostics, and comparison against conformalized alternatives.
- [ ] **A14 — Add post-hoc calibration alternatives.** Deliver isotonic, temperature/scale, and conformal risk-control calibrators as separately configured methods; accept with calibration-only fitting and outer-test reliability metrics.
- [ ] **A15 — Add prediction-time applicability-domain scores.** Deliver descriptor-space distance, ensemble disagreement, and element/chemistry novelty scores for every inference; accept with error-versus-score plots on held-out data and a documented abstention rule.

## B. Training, evaluation, and comparison system

- [ ] **B01 — Implement resumable fold execution.** Deliver per-fold atomic state, completed-fold hashes, and a `resume` command that can continue only an interrupted successor run; accept by interrupting a test run and reproducing the same final artifacts as a clean run.
- [ ] **B02 — Add checkpoint recovery validation.** Deliver checkpoint schema migration/version checks, optimizer-state recovery where relevant, and corruption detection; accept with deliberately truncated, schema-drifted, and byte-altered checkpoint tests.
- [ ] **B03 — Add configuration-driven experiment matrices.** Deliver a declarative matrix for model, featurizer, split, uncertainty method, and compute budget; accept with one generated run plan whose expanded configurations are hash-addressed and non-duplicated.
- [ ] **B04 — Add nested model-selection experiments.** Deliver a successor protocol that selects model class and hyperparameters inside outer-training data only; accept with retained inner-fold choices and no outer-test access prior to final scoring.
- [ ] **B05 — Add a baseline suite runner.** Deliver consistent runners for median, ridge, random forest/ExtraTrees, gradient boosting, kernel method, composition graph model, and any structure model on their valid data subset; accept with one schema-aligned prediction table per method.
- [ ] **B07 — Add repeated group-aware cross-validation.** Deliver repeated canonical-composition-group splits in addition to official Matbench folds; accept with fold manifests, group-isolation proofs, and separate non-leaderboard reporting.
- [ ] **B08 — Add leave-chemistry-family-out evaluation.** Deliver holdouts by anion family, cation family, periodic row/group, and composition complexity; accept with prespecified family definitions, minimum sample sizes, and all failed/undersized families retained.
- [ ] **B09 — Add structure-aware leakage audits.** Deliver duplicate, near-duplicate, reduced-formula, source-paper, and structure/prototype overlap reports where metadata permits; accept with a machine-readable leakage table and per-split effect analysis.
- [ ] **B10 — Add temporal/provenance split evaluation.** Deliver a study based on measurement/publication date or source provenance where reliable fields exist; accept only after coverage, exclusions, and any temporal-label uncertainty are reported.
- [ ] **B11 — Add external-dataset transfer evaluation.** Deliver a pinned and licensed external experimental-gap dataset mapped through explicit unit, composition, duplicate, and provenance rules; accept with train-on-one/test-on-other results and no merged-data claim.
- [ ] **B12 — Add a controlled data-scale study.** Deliver learning curves across fixed training fractions for each principal model; accept with repeated subsamples, confidence intervals, and the complete set of seeds.
- [ ] **B13 — Add feature ablations.** Deliver ablations for fractions, stoichiometry, periodic descriptors, learned embeddings, and structures; accept with identical folds/budgets and paired effect estimates.
- [ ] **B14 — Add architecture ablations.** Deliver controlled comparisons of depth, ensemble members, attention layers, graph-message steps, and fusion gates; accept with a frozen search budget and full trial ledger.
- [ ] **B15 — Add uncertainty-method ablations.** Deliver calibrated comparison of tree dispersion, dropout/ensemble uncertainty, quantiles, and conformal variants; accept with coverage, width, risk-coverage, and target-window ranking impact.
- [ ] **B16 — Add ranking-policy ablations.** Deliver expected improvement, target-window probability, lower-confidence-bound, diversity-aware batch selection, and random/greedy controls; accept with equal budgets, label-blind policy inputs, and exact candidate IDs.
- [ ] **B17 — Add ranking uncertainty intervals.** Deliver bootstrap or randomization confidence intervals for precision, recall, enrichment, and policy differences at every budget; accept with paired comparisons against random and greedy policies.
- [ ] **B18 — Add hypothesis tests with multiplicity control.** Deliver preregistered paired tests or permutation tests across folds/repeats and adjusted confidence intervals; accept with a statistical-analysis script that reads recorded prediction artifacts only.
- [ ] **B19 — Add residual diagnostics.** Deliver residual analyses by measured-gap range, formula complexity, oxide/non-oxide, elemental rarity, family, novelty, and duplicate status; accept with counts, uncertainty intervals, and no empty-bin suppression.
- [ ] **B20 — Add target-noise sensitivity analysis.** Deliver a protocol that models duplicate/measurement disagreement without replacing labels; accept with results across prespecified perturbation levels and original-record traceability.
- [ ] **B21 — Add synthetic sanity batteries.** Deliver controlled datasets that isolate composition scaling, nonlinear interactions, extrapolation, heteroscedastic noise, and duplicate conflicts; accept with expected pass/fail behavior and regression tests.
- [ ] **B22 — Add compute-budget benchmarking.** Deliver wall time, peak RSS, pageouts, artifact size, energy estimate where available, and prediction throughput per model; accept with identical hardware/software receipts and no weakening of a study’s original guard.

## C. Research extensions with materials relevance

- [ ] **C01 — Define application-specific target windows.** Deliver literature-cited, versioned band-gap windows for photovoltaics, photocatalysis, and optoelectronics, each kept as a separate retrospective decision task; accept with no claim that gap alone establishes application fitness.
- [ ] **C03 — Add candidate-diversity-aware ranking.** Deliver batch policies that penalize duplicate formula families and balance elemental/family diversity; accept with selected-set diversity metrics and utility tradeoffs.
- [ ] **C04 — Add synthesis-accessibility filtering as a separate evidence layer.** Deliver a candidate-screening adapter that labels evidence source, availability date, and rule outcome; accept only if the band-gap model output and filter decision remain separately auditable.
- [ ] **C05 — Add stability evidence integration as a separately gated track.** Deliver a provenance-bound stability field with unit/condition definition and missingness report; accept only with explicit rules prohibiting stability inference from the gap model.
- [ ] **C06 — Add toxicity, criticality, and cost annotations.** Deliver an optional composition-derived materials-prior table with source version and uncertainty; accept with element-level citations and no conversion of proxy scores into safety claims.
- [ ] **C07 — Build an out-of-distribution chemistry challenge set.** Deliver a fixed set of rare elements, multinary formulas, uncommon anion families, and composition distances, with no labels used in its construction; accept with test-set manifest and per-stratum metrics.
- [ ] **C08 — Add adversarial formula robustness tests.** Deliver perturbations that preserve composition, alter conventional notation, and introduce malformed/near-valid inputs; accept with invariance where justified and fail-closed behavior otherwise.
- [ ] **C09 — Add label-unit and target-definition robustness.** Deliver auditable eV/meV conversion, censoring/zero-gap policy, and negative-value handling studies; accept with source-label provenance and no silent row deletion.
- [ ] **C10 — Compare alternative objective functions.** Deliver MAE, Huber, quantile, interval-score, and ranking-aware training objectives in a successor study; accept with the same outer splits and no post-result objective selection.
- [ ] **C11 — Add transfer-learning adaptation experiments.** Deliver frozen-versus-finetuned comparisons across external datasets or chemistry domains; accept with source-to-target overlap audit and both zero-shot and adaptation metrics.
- [ ] **C12 — Add prospective simulation with strict reveal order.** Deliver an offline replay where labels are revealed only after candidate selection in each round; accept with immutable reveal logs and a clearly marked simulation status.
- [ ] **C13 — Add experimental-design constraints.** Deliver batch-selection constraints for maximum element count, excluded elements, diversity, and target-window confidence; accept with a feasibility report for every rejected candidate.
- [ ] **C14 — Build a negative-results registry.** Deliver a queryable record of models, descriptor families, splits, and policies that fail to improve a prespecified endpoint; accept with run IDs, artifacts, and no results discarded for underperformance.

## D. Experiment interface and observability

- [ ] **D01 — Expand the CLI into an experiment lifecycle interface.** Deliver `plan`, `run`, `resume`, `compare`, `inspect`, `export`, and `verify` commands with machine-readable JSON output; accept with end-to-end CLI tests against a small fixture study.
- [x] **D02 — Add preflight resource admission.** `ml4materials preflight --output` verifies the data contract, disk, RSS/pageout baseline and writes a durable admission/refusal receipt before allocating a run. Evidence: `artifacts/preflight-20261020.json`.
- [x] **D03 — Add a run registry.** `ml4materials registry` indexes retained receipts and verifies their manifests without modifying them. Evidence: `artifacts/run_registry-20261020.json` and `tests/test_studies.py`.
- [ ] **D04 — Add a live progress monitor.** Deliver phase/fold/model events including elapsed time, RSS, guard state, checkpoint hashes, and failures; accept with a completed event stream from an integration fixture and no hidden exception swallowing.
- [ ] **D05 — Add a comparison dashboard or static report.** Deliver a local interface that filters runs and overlays metrics only when splits/data contracts match; accept with incompatibility warnings and export of the exact comparison table.
- [ ] **D07 — Add candidate review workspace.** Deliver a local table that shows composition, prediction, interval, novelty, provenance, selected-policy rank, and optional evidence annotations; accept with CSV/JSON export retaining source IDs.
- [ ] **D08 — Add schema-validated importers.** Deliver validated CSV/JSON imports for candidate pools, structures, and external labels; accept with row-level error files, duplicate detection, and no silent coercion.

## E. Model inspection and scientific visualization

- [ ] **E01 — Generate out-of-fold parity and residual figures from successful runs.** Deliver vector and raster exports with fold encoding, identity line, units, and data hash; accept only when sourced from complete saved predictions.
- [ ] **E02 — Generate calibration and sharpness figures.** Deliver reliability diagrams, coverage-versus-nominal plots, interval-width distributions, and risk-coverage curves; accept with uncertainty method and split stated in each caption.
- [ ] **E03 — Visualize chemistry-space coverage.** Deliver PCA/UMAP only as an exploratory projection plus quantitative nearest-neighbor distance histograms; accept with train/test/holdout encoding and a fixed random seed.
- [ ] **E04 — Add local explanation reports.** Deliver per-composition feature contribution or attention reports, with stability checks under equivalent formula notation; accept with a warning that explanations describe model behavior rather than causal chemistry.
- [ ] **E06 — Build an error atlas.** Deliver interactive/static panels for the worst residuals, narrow-but-wrong intervals, duplicate conflicts, and OOD cases; accept with direct links to retained row IDs and no target leakage into ranking views.
- [ ] **E07 — Visualize candidate selection trajectories.** Deliver per-policy cumulative hits, precision, diversity, and uncertainty across budget; accept with deterministic candidate order and random-policy confidence bands.
- [ ] **E08 — Generate a publication-figure manifest.** Deliver a manifest mapping each figure/table to run ID, script, source artifact hashes, and caption; accept by regenerating all assets into a clean output directory.

## F. Reproducibility, evidence, and publication outputs

- [ ] **F01 — Package one-command successor experiments.** Deliver a documented command that prepares, preflights, runs, verifies, and renders a named successor study; accept in a clean environment using only pinned inputs and recorded cache locations.
- [ ] **F03 — Preserve full prediction provenance.** Deliver row-level tables containing canonical formula, original IDs, fold, train/calibration membership, model/config hashes, prediction members, intervals, and ranking scores; accept with schema validation and manifest inclusion.
- [ ] **F04 — Add data and artifact retention policies.** Deliver documented immutable run layout, content hashes, retention classes, and migration rules; accept with verification of a completed run and a deliberately modified artifact failure case.
- [ ] **F05 — Add an independent reproduction bundle.** Deliver a minimal archive with source, lockfile, data retrieval instructions/hashes, one fixture run, and verification script; accept when a clean external checkout reproduces its expected fixture manifest.
- [ ] **F06 — Generate complete paper tables from recorded artifacts.** Deliver tables for all baselines, folds, confidence intervals, ablations, holdouts, calibration, ranking, resource use, and failed runs; accept with no manually typed numerical values.
- [ ] **F07 — Produce a statistical-analysis notebook/script.** Deliver a read-only analysis program that loads completed-run artifacts and emits estimates, intervals, tests, and plots; accept with an input-manifest hash and an explicit refusal on incomplete runs.
- [ ] **F09 — Add a results-to-manuscript release gate.** Deliver a check that refuses numerical manuscript rendering unless the corresponding run completed all prescribed folds, controls, and manifest verification; accept with tests for partial and failed run refusal.
- [ ] **F10 — Create a public demonstration.** Deliver a local-first interactive demo or notebook that accepts formulas/candidate CSVs, shows prediction, interval, applicability signals, and claim boundaries; accept with a supplied fixture checkpoint and a visible warning that it is not a stability/synthesis/discovery validator.

## Suggested implementation order

1. First establish an admitted complete `development-v1` receipt (A01) without altering the frozen protocol.
2. Build the successor-study, resumable execution, registry, and preflight substrate (A02, B01–B03, D01–D04, F01–F04).
3. Add scientifically decisive baselines, uncertainty diagnostics, leakage-aware splits, and external transfer tests before advanced architectures (B04–B22, C07–C10, E01–E08, F05–F09).
4. Add composition/structure and multi-task capabilities only as separately scoped data-qualified studies (A03–A15, C01–C14).
5. Publish the candidate-review demo only after it is driven by a successful, manifest-verified run (D05–D08, F10).
