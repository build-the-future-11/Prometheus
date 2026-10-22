# NEUROGENESIS research-feature mega checklist

## Evidence-based starting point

NEUROGENESIS currently contributes a **fail-closed, provenance-bound prequential benchmark** for online regression with delayed labels, distractor shift, sign-conflicting regimes, recurrence, and novelty. Its 20 retained development attempts show a real stability--plasticity tension: delayed SGD (`lr=0.01`) improves mean prequential MSE (0.9484 versus 1.0110 for frozen prediction) but substantially worsens switch and return error. This is valuable benchmark evidence, not evidence for either proposed scientific mechanism.

The project remains `SCOPE_UNRESOLVED`. The historical candidates--a learned update rule and a bounded memory mechanism that can substitute for some parameter updates--are scientifically distinct. No item below authorizes candidate execution or protected evaluation under protocol v1. Items marked **Decision gate** must be completed before their mechanism-specific successors; all other work must preserve v1, its retained negative results, and its evaluation lock.

Each checklist item is an independently reviewable deliverable. Completion requires the stated acceptance evidence, rather than a plan or a passing smoke test.

### Execution evidence recorded 2026-10-20

The retained v1 bundle remains independently `VALID` after this work: 20 verified attempts, 9,600 verified trace rows, zero protected-evaluation runs, and artifact-map SHA-256 `f353206d36fd03ec596a2b4006ef181715c524c43ec13e53d07e2ec28dc089d6`. Derived outputs in `analysis/development-baselines-v1-20261020-r2/` are independently reproducible from those raw artifacts (`analysis_receipts/development-baselines-v1-20261020-r2-verification.json`, `VALID`, seven generated artifacts). The earlier invalid derived-analysis receipt is retained because it exposed and motivated a path-normalization repair in the derived verifier; it is not an experimental failure.

## 1. Scientific contract and protocol succession

- [ ] **[P0, Decision gate] Record one dated owner decision selecting exactly `APPROVED_LEARNED_RULE` or `APPROVED_MEMORY_WEIGHT`.** Define the causal claim, allowed observations at prediction/update time, state permitted across task boundaries, intended use case, and a primary falsifier. **Acceptance:** immutable `decisions/` record validates against a decision schema; `SCOPE_HISTORY.md` gains an append-only row; no existing v1 file is rewritten to imply retroactive approval.

- [ ] **[P0] Publish `protocol_v2.json` as a separately versioned successor.** Freeze candidate identity, task registries, endpoint definitions, budgets, controls, development-selection rules, evaluation admission gates, and falsifier thresholds before an outcome exists. **Acceptance:** strict loader rejects missing, added, duplicate, or changed frozen fields and records the v1 protocol digest as predecessor evidence.

- [ ] **[P0] Specify an information-flow contract for every candidate operation.** Enumerate what `predict`, update/write, retrieval, eviction, consolidation, meta-training, and checkpoint restore can read; explicitly forbid target leakage, future labels, split identity outcomes, and cross-task state carryover unless justified. **Acceptance:** adversarial tests attempt each prohibited route and fail closed with a named contract error.

- [ ] **[P0] Freeze a candidate-specific total-accounting ledger.** Charge prediction, gradient evaluations, parameter updates, memory bytes, retrievals, writes, index probes, evictions, consolidation, meta-training/unrolling, and wall time in common units and per-run counters. **Acceptance:** a bundle verifier recomputes every counter from retained events and rejects a result whose reported totals differ.

- [ ] **[P0] Define the decision rule for development tuning.** State the legal search space, seed budget, selection metric, tie-breaking rule, and how the selected configuration is locked before protected evaluation. **Acceptance:** a signed selection record is generated from development artifacts and evaluation entry points reject any configuration not matching its digest.

- [ ] **[P0] Add a protocol migration validator.** Compare v1 and v2 and classify each changed field as inherited, newly specified, or forbidden historical alteration. **Acceptance:** it produces a machine-readable migration receipt and detects edits to the retained v1 manifest, results, or verification receipt.

- [ ] **[P1] Create a preregistered claim taxonomy.** Map each possible outcome to allowed language: harness correctness, development comparison, held-out generalization, mechanism support, mechanism rejection, or inconclusive. **Acceptance:** result-report generation embeds the strongest supported label and blocks efficacy wording when required gates are absent.

## 2. Candidate model capability -- choose only the approved branch

### Branch A: learned update rule

- [ ] **[P0, conditional on `APPROVED_LEARNED_RULE`] Implement a differentiable learned-update-rule policy behind `LearnedRuleCandidate`.** Its update network consumes only the frozen legal local signals and emits a bounded parameter update. **Acceptance:** a deterministic mini-task changes weights only after delayed labels arrive; snapshots restore bitwise-identical continuation predictions.

- [ ] **[P0, conditional] Implement the meta-training runner for registered `meta_train` tasks.** Support truncated unrolling with explicitly counted unroll length, outer updates, auxiliary-network parameters, and retained optimizer state. **Acceptance:** one command emits a meta-training manifest, per-outer-step trace, final candidate checkpoint, and accounting receipt bound to source/config hashes.

- [ ] **[P0, conditional] Add a mechanism-off counterpart with identical base predictor and legal features.** It must replace the learned updater with the selected conventional optimizer without removing accounting or changing task timing. **Acceptance:** a feature-parity test confirms identical inputs and a trace-diff test isolates only the update decision.

- [ ] **[P1, conditional] Implement update parameterization variants.** Compare direct delta, learning-rate modulation, coordinate-wise gating, and normalized-gradient transforms under the same output norm limit. **Acceptance:** each variant has a frozen config, event trace, and ablation table including total meta-training and inference cost.

- [ ] **[P1, conditional] Add learned-rule safety guards.** Bound update norm, detect NaN/Inf in inner or outer computation, and preserve a failure artifact rather than retrying a seed. **Acceptance:** injected unstable gradients yield a retained `FAILED` attempt with the first invalid step and no silent clipping beyond the declared rule.

- [ ] **[P1, conditional] Test meta-objective alignment.** Measure whether optimizing the learned objective improves the external prequential MSE rather than only the surrogate loss. **Acceptance:** recorded correlation/disagreement analysis across development runs and an explicit failure classification for objective improvement without endpoint improvement.

### Branch B: bounded memory-weight adaptation

- [ ] **[P0, conditional on `APPROVED_MEMORY_WEIGHT`] Implement a bounded episodic memory policy behind `MemoryWeightCandidate`.** Store only legal delayed-label examples or derived keys/values and make prediction before current-target access. **Acceptance:** a timeline test proves each memory record is written no earlier than `label_available_at` and an exact-capacity test never exceeds the frozen byte limit.

- [ ] **[P0, conditional] Implement a retrieval scoring and read path.** Define the key representation, similarity/distance, deterministic tie-breaking, top-k, and how retrieved values affect prediction or weight updates. **Acceptance:** a synthetic fixture produces known rankings and a trace records queried keys, selected record IDs, scores, and contribution to the prediction.

- [ ] **[P0, conditional] Implement explicit write admission and eviction.** Compare utility-aware admission against fixed FIFO/LRU only after the rules are frozen. **Acceptance:** capacity-pressure tests retain exactly the documented records, log every admission/rejection/eviction, and reconcile event counts with resource usage.

- [ ] **[P1, conditional] Implement memory consolidation.** Define when records may be merged, compressed, or converted into weights and whether originals remain recoverable. **Acceptance:** consolidation events are replayable from the event log, preserve the declared statistic within tolerance, and consume the declared compute/storage budget.

- [ ] **[P1, conditional] Implement a memory-to-weight gating mechanism.** Let the policy select memory retrieval, a parameter update, both, or neither under a frozen decision rule. **Acceptance:** every step records the selected adaptation action and an ablation can force each action class while preserving information and cost accounting.

- [ ] **[P1, conditional] Add reset, contamination, and task-boundary controls.** Test cold reset, permitted within-task state, and prohibited state carryover separately. **Acceptance:** a fresh task reproduces its cold-start trace after any prior task, unless a protocol field explicitly authorizes and logs transfer state.

- [ ] **[P1, conditional] Add cache/replay controls matched for bytes and reads.** Include reservoir replay, FIFO cache, LRU cache, nearest-neighbor retrieval, and ordinary delayed updates as relevant. **Acceptance:** control manifests match capacity, retrieval/write limits, and label timing; the verifier rejects unmatched comparisons.

## 3. Training, execution, and recovery

- [ ] **[P0] Generalize the runner from fixed baselines to a protocol-admitted policy registry.** Policies declare capabilities, required budget dimensions, checkpoint schema, and legal split(s). **Acceptance:** unsupported policy/split combinations fail before task generation and registered policies produce uniform attempt artifacts.

- [ ] **[P0] Add interruption-safe checkpoint and resume at the event boundary.** Persist policy state, pending labels, RNG state, resource usage, trace hash prefix, and completed step. **Acceptance:** an injected interruption at multiple steps resumes to the same final result and trace SHA-256 as uninterrupted execution.

- [ ] **[P0] Make every attempt idempotent and collision-safe.** Attempt IDs must bind task, policy, config digest, source digest, and replicate; partially written outputs must be detected. **Acceptance:** duplicate launch attaches to the exact completed receipt or fails with a clear collision error; no attempt directory is overwritten.

- [ ] **[P0] Add a matrix scheduler with deterministic serial and bounded-parallel modes.** Preserve a stable attempt order and make concurrency an explicitly recorded execution property. **Acceptance:** serial and allowed-parallel replays yield identical per-attempt metrics and trace hashes for deterministic policies.

- [ ] **[P1] Implement a declarative hyperparameter sweep compiler.** Expand a frozen search specification into named configurations without sampling from protected evaluation. **Acceptance:** emitted configs have canonical JSON digests, no duplicates, and a search-budget receipt showing every requested candidate.

- [ ] **[P1] Add early termination only for predeclared safety criteria.** Terminate on resource violation, non-finite values, or a frozen instability threshold while retaining the partial trace. **Acceptance:** test runs demonstrate each termination reason, report the same denominator rules in aggregation, and never silently drop terminated attempts.

- [ ] **[P1] Add capability-aware checkpoint conversion.** Reject restoring a checkpoint into a policy with different feature dimension, candidate identity, memory capacity, or optimizer schema. **Acceptance:** negative compatibility fixtures fail with field-level diagnostics; compatible snapshots produce exact continuation equivalence.

- [ ] **[P2] Package a CPU-only reference execution environment.** Pin interpreter constraints and run the benchmark with no network or third-party runtime dependencies unless a successor explicitly adds them. **Acceptance:** a clean-environment command builds, validates, runs a small matrix, aggregates it, and verifies its bundle.

## 4. Evaluation suite and robust baselines

- [ ] **[P0] Implement a candidate-specific development/evaluation manifest schema.** Register task-family parameters, split commitments, replications, policy configs, and statistical plan. **Acceptance:** duplicate identities across any split, undeclared task-family parameters, and evaluation edits after admission are rejected.

- [ ] **[P0] Add a tuned conventional optimizer baseline suite.** Include a legal grid for SGD, Adam, and any justified online second-order or adaptive baseline, selected only on development data. **Acceptance:** every candidate comparison names the locked winning conventional baseline, its full sweep denominator, and the selection receipt.

- [ ] **[P0] Add resource-matched ablation runners.** Equalize predicted labels, feature inputs, wall-time ceiling, state bytes, update counts, and memory operations as appropriate to the approved mechanism. **Acceptance:** a comparator emits a resource-difference report and fails an ablation whose required matching dimension differs.

- [ ] **[P1] Add delayed-label stress curves.** Sweep label delay from immediate to horizon-scale delays while holding generated streams and budget rules fixed. **Acceptance:** each method receives identical task identities per delay; output includes error, failure, and resource curves with confidence intervals.

- [ ] **[P1] Add nonstationarity severity sweeps.** Vary conflict angle, switch frequency, recurrence interval, novelty angle, noise, and distractor variance independently. **Acceptance:** a factorial manifest records every cell and produces a method-by-severity response surface with missing/failed cells retained.

- [ ] **[P1] Add capacity and compute scaling studies.** Sweep state-byte budgets, memory slots, retrieval top-k, update budget, and horizon length. **Acceptance:** plots report the full error--bytes--operations Pareto frontier rather than a single favorable budget.

- [ ] **[P1] Add out-of-family synthetic transfer tasks.** Implement nonlinear targets, sparse informative coordinates, changing feature scale, correlated distractors, abrupt/gradual drift, and recurrent regimes with new periods. **Acceptance:** held-out family definitions and seeds are committed before development selection; evaluation results retain all cells.

- [ ] **[P1] Add a real-stream adapter with a documented delayed-feedback source.** Use a public, legally distributable temporal regression or forecasting stream whose ordering and label delay can be reconstructed. **Acceptance:** an immutable dataset manifest records version, checksum, temporal split, preprocessing, and a baseline-only dry run reproduces on a clean environment.

- [ ] **[P1] Add adversarial leakage and confounding tests.** Perturb future-label availability, task ordering, regime metadata, and split identifiers to prove the policy cannot exploit them. **Acceptance:** each injected channel is either inaccessible or raises a contract error; a deliberately leaky reference policy is detected by the audit.

- [ ] **[P2] Add missingness and delayed/out-of-order feedback robustness.** Model missing labels, censored labels, unequal delays, and delayed feedback bursts without reordering predictions. **Acceptance:** traces expose availability events and result tables separate performance from the fraction of labels actually delivered.

- [ ] **[P2] Add initialization and scale transfer experiments.** Evaluate new feature dimensions, parameter scales, and zero/nonzero initializations without retuning on the target condition. **Acceptance:** transfer manifest identifies source versus target conditions and aggregates per-condition failure rates and effect sizes.

## 5. Analysis, model inspection, and falsification

- [x] **[P0] Generate per-run prequential learning-curve figures.** `scripts/analyze_bundle.py` generates both the policy-mean and all-20-attempt rolling-MSE SVGs in `analysis/development-baselines-v1-20261020-r2/figures/`; individual panels preserve seed identities, regime boundaries at 120, 240, and 360, and the input-artifact-map digest. **Evidence:** `analysis_receipts/development-baselines-v1-20261020-r2-verification.json` regenerated all seven outputs byte-for-byte.

- [x] **[P0] Generate a stability--plasticity diagnostic.** `analysis/development-baselines-v1-20261020-r2/figures/stability_plasticity.svg` plots mean first-24 switch versus return-regime MSE for all four policies; the generated CSV/LaTeX tables include operations and state bytes. **Evidence:** it reproduces the recorded SGD 0.01/frozen tradeoff and is hash-verified by the derived-analysis receipt.

- [ ] **[P0, conditional] Add mechanism event visualizations.** For learned rules show update magnitude/direction and gate activations; for memory show writes, retrievals, scores, occupancy, evictions, and consolidation lineage. **Acceptance:** every marker links to a retained event record and a replay script regenerates the visualization without model execution.

- [ ] **[P1] Add representation and parameter trajectory inspection.** Track weight vectors, update cosine similarity, effective learning rates, memory-key geometry, or updater hidden state as appropriate. **Acceptance:** dimensionality reduction choices are fixed in config and raw coordinates/checkpoints are exported alongside plots.

- [ ] **[P1] Add error decomposition by regime and label age.** Report performance for initial, conflict, return, and novel regimes; bucket errors by time since last feedback and availability delay. **Acceptance:** aggregation verifies bucket counts sum to all predictions and exposes undefined buckets rather than imputing values.

- [ ] **[P1] Add counterfactual trace replay.** Re-evaluate a frozen policy state on alternative legal action choices or with specific memory events masked, without exposing future labels. **Acceptance:** replay records provenance, never mutates the original result, and quantifies the endpoint change attributable to each masked operation.

- [ ] **[P1] Implement unit tests for scientific falsifiers.** Encode the expected failure of a no-op candidate, a label-leaking candidate, an over-capacity memory, and an unstable learned updater. **Acceptance:** each intentionally defective reference is rejected for the declared scientific reason, independent of raw MSE.

- [ ] **[P2] Add calibration and uncertainty diagnostics when predictions include uncertainty.** Evaluate interval coverage, sharpness, and regime-conditional residual calibration. **Acceptance:** uncertainty claims are omitted unless the policy provides a documented predictive distribution and the calibration table is generated from held-out traces.

## 6. Statistical evidence and paper-ready outputs

- [ ] **[P0] Implement paired statistical analysis at the task/seed level.** Use a predeclared paired estimator, confidence interval method, effect size, and multiplicity treatment across endpoints and conditions. **Acceptance:** analysis consumes complete retained matrices, lists all excluded values with reasons, and refuses to calculate a comparison with mismatched identities.

- [x] **[P0] Produce a complete run-accounting table.** `analysis.json` records requested 20, started 20, completed 20, failed 0, terminated 0, resumed 0, excluded 0, and protected-evaluation runs 0. **Evidence:** values are recomputed from `run_manifest.json` and `receipt.json`, then checked by `scripts/verify_analysis.py`.

- [x] **[P0] Build a paper-results generator.** `scripts/analyze_bundle.py` emits CSV, LaTeX, SVG, JSON, and an offline HTML report directly from the retained bundle; `scripts/assemble_submission.py` emits the evidence-bound LaTeX paper. **Evidence:** the generated paper compiled successfully with Tectonic, and its table values are derived from the verified analysis artifact.

- [ ] **[P1] Generate ablation tables with cost columns.** Report endpoint, switch/return error, state bytes, reads/writes, updates, gradient evaluations, meta-training cost, wall time, and failure count. **Acceptance:** the generator rejects an ablation table missing any cost dimension applicable to the selected mechanism.

- [ ] **[P1] Add robustness and transfer figure panels.** Produce preregistered multi-condition plots with all seeds/cells visible or summarized by a documented rule. **Acceptance:** panel manifests encode exact input bundles and plots mark unrun, failed, and not-applicable cells distinctly.

- [ ] **[P1] Write an evidence-bound limitations section generator.** Populate limitations from protocol metadata: synthetic/real scope, data breadth, compute bounds, untested conditions, split status, and independent-reproduction status. **Acceptance:** generated prose cannot state protected evaluation or independent replication occurred unless corresponding receipts exist.

- [ ] **[P2] Assemble a publication artifact bundle.** Package manuscript sources, exact configs, source snapshot, environment record, figures, tables, raw aggregates, and verifier receipt. **Acceptance:** archive manifest hashes every member and an external unpack-and-verify command succeeds without altering the original results.

## 7. Experiment interface, reproducibility, and public demonstration

- [ ] **[P0] Extend the CLI with `plan`, `run`, `resume`, `verify`, `analyze`, and `report` commands.** Each command must accept explicit paths and print a canonical artifact ID. **Acceptance:** CLI integration tests cover a full small experiment lifecycle and invalid split/candidate requests fail before writes.

- [x] **[P1] Add a read-only experiment dashboard or static report.** `analysis/development-baselines-v1-20261020-r2/report.html` is an offline report with the baseline table, three vector figures, claim boundary, and input-artifact-map digest. It has no run controls. **Evidence:** it is generated and byte-verified from retained inputs.

- [ ] **[P1] Add configuration composition with immutable resolved configs.** Permit concise base/override experiment specs while writing the fully resolved canonical JSON before execution. **Acceptance:** the same inputs resolve to the same digest; conflicting overrides and undeclared keys fail validation.

- [x] **[P1] Add a run-comparison export.** The analysis bundle exports `development_baselines.csv`, `analysis.json`, `development_baselines.tex`, and `report.html`, all bearing the development-only claim boundary and source-artifact map. **Evidence:** `scripts/verify_analysis.py` regenerates and compares every derived export byte-for-byte.

- [ ] **[P1] Create one-command reproduction targets.** Provide commands for unit tests, a tiny deterministic smoke matrix, full admitted development rerun, bundle verification, analysis, and paper figures. **Acceptance:** each target starts from a clean output directory, never overwrites retained evidence, and returns a manifest path.

- [ ] **[P1] Record deterministic execution metadata.** Persist source tree digest, protocol/config digests, interpreter/platform, RNG seeds, dependency lock, invocation, and hardware-relevant limits. **Acceptance:** verifier detects altered source/config/environment metadata and reports whether exact replay is expected or only numerical tolerance is applicable.

- [ ] **[P2] Add content-addressed artifact storage and retention policy.** Deduplicate immutable traces/checkpoints while preserving attempt-level provenance and failed outputs. **Acceptance:** garbage-collection dry run reports only unreferenced artifacts; a restored bundle passes the independent verifier.

- [ ] **[P2] Create a public interactive demonstration.** Let users choose a precomputed approved experiment and inspect delayed-label timing, regime shifts, predictions, resource use, and candidate events; never present a toy animation as new efficacy evidence. **Acceptance:** demo assets are generated from a verified bundle, display its claim boundary and artifact hashes, and offer a downloadable trace/config pair.

- [ ] **[P2] Publish an inference/replay notebook.** Walk through one retained task stream, reconstruct predictions step by step, expose legal information at each timestep, and recreate the inspection figures. **Acceptance:** notebook executes from a clean environment using only published artifacts and checks expected trace hashes before rendering.

## Completion standard

The research system is ready to make a candidate-specific claim only when the selected branch has: (1) an owner-approved v2 protocol, (2) a locked development selection, (3) complete matched controls and ablations, (4) verified candidate development artifacts, (5) explicit owner admission for protected evaluation, (6) complete held-out evaluation artifacts, and (7) regenerated statistical and paper outputs. Until then, the strongest honest result remains the current validated baseline harness and its documented stability--plasticity tradeoff.
