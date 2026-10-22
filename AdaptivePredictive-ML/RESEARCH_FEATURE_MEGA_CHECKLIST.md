# Atlas Research Engine — Research Feature Mega Checklist

## Evidence-bound project reading

**Inspection receipt (2026-10-20).** This `AdaptivePredictive-ML` directory is empty
and untracked. Its parent repository, `/Volumes/PRO-BLADE/Atlas`, has tracked source
files deleted from the current working tree; those user changes were not altered.
The checklist below is therefore based read-only on committed snapshot
`21b9338f23df46b32af734cb33903528a1899e8a` (2026-09-17), not on a claimed working
implementation or recorded research result.

**Core contribution currently evidenced.** Atlas is a deterministic, evidence-safe
research orchestration layer: it compiles a question into a hypothesis, fixed
experiment matrix, stopping rule, and required artifacts; exports a fail-closed task
bundle to Percy; validates evidence/claims; and includes a frozen LAM-JEPA ARC-v3
reproduction adapter. The committed README explicitly limits the milestone to `v0.1`.
The LAM-JEPA adapter retains `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`; no positive
model claim or outcome-bearing run is present in this workspace.

**Meaningful capability gap.** Atlas plans and validates small fixed matrices, but it
does not yet form a complete research system that can express rich study designs,
execute and recover workloads, ingest immutable evidence from heterogeneous executors,
perform appropriate statistical inference, make claim-to-evidence traceability
inspectable, or generate paper-ready evidence from completed studies. The items below
close those specific gaps. Each checked item must produce the named artifact and pass
the stated acceptance evidence; no UI, passing unit test, or task completion alone
constitutes scientific validation.

## 1. Protocol compiler and scientific-contract capabilities

- [ ] **Version a typed study-protocol schema.** Define questions, estimands, units,
  interventions, outcomes, datasets, splits, baselines, exclusions, seed policy,
  stopping rules, and planned analyses in `atlas.protocol.v1`; publish JSON Schema,
  migration fixtures, and rejection tests for ambiguous protocols.

- [ ] **Compile explicit estimands, not free-text metrics.** Represent the comparison
  population, treatment/control contrast, aggregation, horizon/window, and target
  direction; emit a machine-readable estimand card and test it against a multi-seed
  fixture where an apparently similar aggregate gives a different answer.

- [ ] **Add a protocol-lint pass for scientific defects.** Reject missing baselines,
  unpaired treatment/control seeds, threshold-less success criteria, undefined split
  roles, incompatible metric directions, and test-data use; deliver diagnostic codes
  with source locations and a negative fixture for every code.

- [ ] **Support preregistered multi-stage studies.** Model exploration, confirmation,
  replication, and follow-up as distinct stages with promotion criteria and no
  retrospective mutation; output a stage DAG and prove an exploratory result cannot be
  exported as confirmatory evidence.

- [ ] **Add protocol amendment semantics.** Create a superseding-protocol record that
  preserves the immutable predecessor, precise field diff, rationale, author/time, and
  changed hypothesis; test that changing an estimand or matrix always changes identity.

- [ ] **Implement protocol signatures and canonical digests.** Canonicalize the full
  scientific contract (including analysis plan, not creation time) and sign it with a
  configured key; verify tamper detection and publish a portable verification command.

- [ ] **Compile factorial and nested experiment designs.** Support factors, levels,
  blocked/randomized assignments, repeated measures, and nested CV without exploding
  accidental Cartesian products; export a design table with planned denominators.

- [ ] **Add adaptive-design guardrails.** Permit sequential or Bayesian optimization
  only when the update rule, look schedule, alpha/decision allocation, and stopping
  boundary are frozen; generate a simulation demonstrating false-positive control.

- [ ] **Introduce explicit data-governance contracts.** Bind licensing, consent/use
  restrictions, PII handling, retention, permitted splits, and forbidden artifacts to a
  study; fail planning and execution if a declared use conflicts with the contract.

- [ ] **Build a reusable perturbation/robustness DSL.** Express distribution shift,
  corruption, missingness, label noise, time drift, subgroup, and resource constraints;
  compile each condition into a deterministic, hashable transformation recipe.

- [ ] **Add a protocol feasibility estimator.** Calculate planned task count, GPU/CPU
  hours, storage, expected artifact volume, and required data availability from the
  design; generate an admission report before any external execution is authorized.

- [ ] **Create a domain reference protocol for adaptive prediction.** Add a complete
  non-stationary forecasting/latent-prediction protocol with rolling temporal splits,
  delayed labels, calibration, forgetting, shift tests, and matched baselines; use it as
  an executable integration fixture, not evidence of performance.

## 2. Execution, recovery, and evaluator capability

- [ ] **Define an executor-neutral task lifecycle.** Specify durable task states,
  attempts, cancellation, retry classification, heartbeat, timeout, and terminal
  evidence requirements; publish a state-transition table and property tests rejecting
  impossible transitions.

- [ ] **Implement a local reference executor.** Run an exported task graph with
  isolated working/artifact directories, captured environment, stdout/stderr, exit
  status, and resource observations; reproduce a fixture study without Percy.

- [ ] **Add idempotent task recovery.** Resume after process death without rerunning a
  verified attempt or overwriting artifacts; demonstrate an intentional mid-matrix
  interruption whose resumed evidence ledger has exactly the planned task identities.

- [ ] **Add container/environment realization.** Bind every task to a lockfile plus
  OCI image or reproducible environment specification; record image digest and package
  inventory, then verify a fresh host reproduces the fixture evaluator output.

- [ ] **Implement resource enforcement and reporting.** Enforce declared runtime,
  memory, GPU, CPU, and storage limits; emit per-attempt peak usage and classify
  resource-limit termination separately from scientific negative outcomes.

- [ ] **Add executor adapters behind one contract.** Implement and contract-test local,
  Percy, and one batch/HPC adapter, preserving task ID, protocol digest, attempt record,
  resource request, and artifact contract across exports/imports.

- [ ] **Create a remote-status reconciliation job.** Reconcile executor statuses with
  Atlas attempt records using stable IDs and append-only receipts; flag missing, stale,
  duplicate, or contradictory status updates rather than guessing completion.

- [ ] **Implement dependency-aware partial-failure policy.** Encode which downstream
  analyses are blocked versus allowed by a failed cell and require a denominator-aware
  classification; test that a parent claim cannot silently omit failed prerequisite
  experiments.

- [ ] **Add checkpoint lineage to task contracts.** Require input checkpoint hashes,
  checkpoint-producing steps, resume compatibility metadata, and output hashes; reject
  a resume from a checkpoint made by a different protocol or data split.

- [ ] **Support human approval gates as evidence objects.** For costly, safety-relevant,
  or external-data steps, record the exact pending action, resolved inputs, approver,
  decision, and timestamp; make execution refuse to cross an unapproved gate.

- [ ] **Build an evaluation-plugin API.** Let a workload register metric extraction,
  validation, aggregation, required diagnostics, and evaluator version while keeping the
  evaluator isolated from task code; ship conformance tests and a minimal plugin.

- [ ] **Implement streaming/online evaluation.** Score predictions before labels are
  revealed, preserve label-arrival time, and partition prequential metrics by regime;
  validate ordering on a synthetic delayed-label stream.

## 3. Evidence, provenance, and claim evaluation

- [ ] **Version the evidence ledger schema.** Store protocol digest, task/attempt IDs,
  source state, data and environment hashes, artifact digests, measurements, validation
  status, and immutable timestamps in append-only JSONL/SQLite; supply a schema
  migration and integrity verifier.

- [ ] **Make evidence content-addressable.** Store artifacts by SHA-256 (or stronger),
  deduplicate safely, and retain logical names as references; prove altered bytes cannot
  satisfy a prior artifact receipt.

- [ ] **Capture source provenance beyond commit SHA.** Record repository remote, commit,
  clean/dirty diff hash, submodule state, untracked-file policy, and generated-source
  provenance; flag a mismatch at evaluator time.

- [ ] **Create dataset/split lineage records.** Bind every training, validation, test,
  and synthetic dataset to manifest/hash, transform version, row/entity/time IDs, and
  split membership; produce a leakage report for each completed study.

- [ ] **Add artifact authenticity verification.** Support signed remote uploads and
  trusted-executor attestations, retain raw receipts, and classify unsigned or
  unverifiable artifacts as untrusted rather than complete evidence.

- [ ] **Build an evidence ingestion quarantine.** Validate media type, schema, size,
  path containment, duplicate JSON keys, archive traversal, hash, and signature before
  promoting an executor artifact; keep rejected inputs with a reason receipt.

- [ ] **Support structured metric provenance.** Store each value with metric definition,
  evaluator version, slice, denominator, units, uncertainty/CI, source artifact byte
  range or query, and run identity; reject bare scalar metrics for confirmatory claims.

- [ ] **Extend claim classification for uncertainty.** Add confidence/credible interval,
  practical-effect threshold, multiplicity adjustment, and missing-data policy to claim
  specs; test supported, unsupported, inconclusive, and invalid-evidence boundaries.

- [ ] **Implement paired statistical analysis plugins.** Provide block bootstrap,
  paired permutation, and hierarchical/cluster-aware options with precondition checks;
  emit effect sizes, intervals, assumptions, and seed-resolved inputs rather than only
  a p-value.

- [ ] **Create a claim-evidence graph.** Link manuscript claim → estimand → analysis →
  metrics → artifacts → attempts → protocol/data/source hashes; export JSON-LD and an
  SVG/HTML trace view that marks missing or invalid links.

- [ ] **Add negative and null-result preservation.** Classify preregistered failures,
  null effects, resource failures, and protocol violations distinctly; make summary
  exporters include their denominators and prohibit deleting them from a study view.

- [ ] **Build a reproducibility replay verifier.** Re-run evaluator/analysis code from
  a frozen evidence bundle and compare all derived tables/figures to hashed originals;
  produce a diff report for expected floating-point tolerances versus material drift.

## 4. Research extensions and scientifically motivated studies

- [ ] **Generalize the benchmark subsystem beyond exact match.** Add regression,
  ranking, probabilistic, generation, and structured-output metric adapters with locked
  normalizers and adversarial scoring fixtures for each metric family.

- [ ] **Create a benchmark-card and contamination contract.** Record task construct,
  source/labels, license, known biases, split policy, annotation quality, intended use,
  and contamination audit; block a published comparison without these fields.

- [ ] **Implement cross-dataset replication plans.** Define a common estimand while
  allowing dataset-specific preprocessing and report per-dataset plus meta-analytic
  effects; demonstrate with at least two small licensed fixture datasets.

- [ ] **Add robustness-to-protocol analysis.** Systematically vary defensible
  preprocessing, metric, seed, hyperparameter, and exclusion choices specified as a
  multiverse; export a specification-curve figure tied to individual run IDs.

- [ ] **Implement baseline qualification records.** Require baseline source/version,
  known tuning budget, implementation validation, and why it is relevant; prevent
  comparison to a baseline whose protocol is not equivalently resourced.

- [ ] **Add hyperparameter-search accounting.** Track all trials, data used for
  selection, compute, failed trials, and final-selection rule; prohibit test-split
  selection and emit a tuning-budget comparison table.

- [ ] **Support causal/interventional study templates.** Add treatment assignment,
  covariate balance, interference/positivity assumptions, estimand, and sensitivity
  analyses; lint observational claims that are worded as causal without this contract.

- [ ] **Implement temporal and distribution-shift study templates.** Add rolling-origin
  evaluation, temporal holdout, event/change-point annotations, deployment-delay
  simulation, and drift slices; ship fixtures that catch random-split leakage.

- [ ] **Add fairness/subgroup analysis contracts.** Define protected/proxy fields,
  subgroup eligibility thresholds, metric slices, uncertainty, missingness policy, and
  privacy constraints; report insufficient-data groups rather than suppressing them.

- [ ] **Build an ablation planner.** Compile component-removal/alternative studies with
  capacity-, compute-, and tuning-budget-matched controls; automatically reject an
  ablation where the removed component changes an unaccounted resource budget.

- [ ] **Create a mechanism/representation-probe protocol.** For models with latent
  state, attention, retrieval, or memory, register a falsifiable probe, held-out
  intervention, counterexample, and causal limitation; save probe datasets and results
  as ordinary evidence cells.

- [ ] **Add a research-risk register.** Track anticipated leakage, benchmark saturation,
  weak controls, resource bias, test access, and invalid-inference risks with owner,
  trigger, mitigation, and protocol impact; expose unresolved high risks in exports.

## 5. Experiment interface and inspection

- [ ] **Deliver a protocol-authoring CLI.** Support `atlas protocol init|lint|compile|
  amend|sign` with inline schema errors, dry-run design table, and canonical JSON output;
  exercise every command in an end-to-end shell test.

- [ ] **Deliver an execution CLI.** Support `submit`, `status`, `resume`, `cancel`,
  `ingest`, `verify`, and `replay` against the common lifecycle API; return structured
  JSON and nonzero, documented exit codes for every blocked/invalid state.

- [ ] **Build a study dashboard from the ledger.** Show protocol identity, design
  matrix, cell/attempt state, resource consumption, evidence completeness, outcomes,
  and high-risk flags directly from recorded data; include an empty/invalid-evidence
  state that makes absence visible.

- [ ] **Add a run-comparison workspace.** Select compatible studies and compare
  protocol diffs, baseline versions, data/split hashes, distributions of seed-level
  metrics, CIs, resource cost, and claim classifications without manual spreadsheets.

- [ ] **Build a timeline inspector.** Render question/protocol creation, amendments,
  task attempts, artifacts, evaluator runs, and claim changes on one immutable
  chronological trace; link every event to its raw record.

- [ ] **Build an artifact/evidence inspector.** Allow drill-down from a metric to its
  parsed source artifact, hash, validation checks, producing command, environment, and
  logs; redact or gate protected artifacts according to the data contract.

- [ ] **Add a design-matrix inspector.** Visualize planned versus completed cells by
  factor, seed, perturbation, and stage, including failed/missing cells and denominators;
  export the matrix as CSV and a provenance-bound static figure.

- [ ] **Add model-output inspection hooks.** Standardize optional prediction tables,
  error slices, calibration curves, representation probes, and attention/memory traces
  as evaluator artifacts; provide a viewer that labels these as diagnostic, not proof.

- [ ] **Create a failure-forensics view.** For every failed/invalid task, show
  classified cause, raw logs, resource profile, last checkpoint, protocol/data identity,
  and downstream blocked claims; export a machine-readable incident report.

- [ ] **Implement role-based redaction and access audit.** Separate public metadata,
  private raw artifacts, and credential-bearing execution details; test that public
  exports cannot expose protected paths, values, or tokens.

## 6. Reproducibility, paper-ready outputs, and demonstration

- [ ] **Create a one-command reference study.** From a fresh environment, compile,
  execute locally, ingest, verify, analyze, and render a small synthetic adaptive-
  prediction study; emit the complete evidence bundle and expected receipt hashes.

- [ ] **Publish a reproducibility bundle format.** Package protocol, source snapshot,
  environment lock/image digest, data or generator, split manifest, task receipts,
  evidence ledger, evaluator, analysis, and artifacts; verify it offline in CI.

- [ ] **Add deterministic-control reporting.** Record RNG, framework determinism flags,
  hardware, precision, concurrency, and nondeterministic operations; perform a repeated
  reference run and report exact/tolerance-bound differences per derived quantity.

- [ ] **Generate provenance-bound paper tables.** Render baseline, robustness,
  ablation, resource, and failure/denominator tables from locked query definitions; put
  protocol/run IDs, estimator, CI, and export hash in each table caption or footnote.

- [ ] **Generate provenance-bound figures.** Render effect/CI, seed distribution,
  robustness, design completion, and evidence-lineage figures from ledger queries;
  include a sidecar JSON that recreates each figure from its run and artifact IDs.

- [ ] **Create a manuscript claim gate.** Require every declarative result sentence to
  reference a claim ID and fail the export when evidence is incomplete, invalid,
  exploratory-only, or contradicted; output a claim coverage and unresolved-limitations
  report.

- [ ] **Write an auto-generated limitations section.** Synthesize only recorded bounds:
  incomplete cells, failed replications, data/shift coverage, resource constraints,
  protocol amendments, and assumption violations; allow editorial wording but retain
  links to the ledger evidence.

- [ ] **Ship an interactive public demonstration.** Provide a browser or local app that
  replays the synthetic reference study from bundled data, lets users inspect the
  protocol/DAG/evidence/claim graph, and visibly distinguishes planned, executed, and
  scientifically supported states.

- [ ] **Ship an inference/workload demonstration.** Package a CLI or notebook that
  runs a licensed toy adaptive-prediction workload, exports standard Atlas artifacts,
  and displays forecast/error/shift diagnostics without claiming real-world efficacy.

- [ ] **Create release qualification and archival receipts.** Verify the release bundle
  in a clean environment, archive immutable artifacts with checksums and retention
  policy, issue an SBOM/license report, and publish a release status that separates
  engineering completeness from independent scientific replication.

## Completion standard

The contribution becomes experimentally substantiated only when an immutable protocol
and complete evidence ledger support the relevant claims through matched baselines,
all planned cells (including negative/failed ones), appropriate analysis, robust
replication, and reproducible paper/demo outputs. Until then, Atlas is an
engineering/planning system with a frozen negative-or-inconclusive LAM-JEPA adapter—not
validated evidence that an adaptive predictive model works.
