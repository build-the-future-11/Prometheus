# Atlas Research Engine — Research Feature Mega Checklist

## Evidence-grounded scope

Atlas is an **evidence-bound research execution system**: it turns a falsifiable
question into a frozen run manifest, a matrix of treatment/control experiments,
durable executor tasks, retained evidence, and a claim classification that fails
closed when evidence is incomplete. The committed v0.1 implementation already
has deterministic planning, evidence hashing, strict `atlas.percy-tasks.v0.1`
export verification, a tiny exact-match benchmark utility, and a frozen
LAM-JEPA ARC-v3 reproduction adapter. A later reachable branch adds a
pre-outcome Weather-JEPA protocol freeze.

The repository contains no recorded outcome-bearing Atlas study result. Thus,
none of the work below may be represented as support for LAM-JEPA,
Weather-JEPA, or any other model until the corresponding preregistered study is
executed and independently verified. Frozen protocols remain immutable;
changed hypotheses require versioned successor protocols.

Every item below has a tangible research-system deliverable and a completion
receipt. “Done” means the listed acceptance evidence exists, not merely that
the code compiles.

## A. Core scientific and execution model

| ID | Deliverable | Acceptance evidence |
|---|---|---|
| A01 | Add a versioned `StudyProtocol` schema that binds question, population, unit of analysis, primary/secondary endpoints, estimand, baseline definitions, seed policy, stopping rule, and test-split policy. | A schema-valid protocol JSON round-trips without loss; changing any scientific field changes the protocol digest. |
| A02 | Replace the single global `Criterion` with endpoint specifications supporting metric direction, aggregation, confidence interval method, practical-effect threshold, and missing-data policy. | A fixture with primary and secondary endpoints yields the frozen classifications and rejects an undeclared endpoint. |
| A03 | Add explicit control-arm, comparator-version, and mechanism-off fields to each experiment rather than free-text treatment/control labels. | The planner refuses a proposed-method experiment without a declared matched control and mechanism-off comparator. |
| A04 | Implement a typed experiment graph: data preparation, training, calibration, evaluation, verifier, and aggregation nodes with declared dependencies and inputs/outputs. | A graph export is a DAG, has deterministic node IDs, and rejects cycles or unbound inputs. |
| A05 | Add content-addressed input declarations for source trees, datasets, split files, containers/environments, and hyperparameter files. | A run cannot enter execution when an input digest differs from its protocol-bound digest. |
| A06 | Implement protocol amendment records that preserve the parent digest, state the reason and date, and create a new successor ID instead of editing an executed protocol. | Attempting to amend an executed protocol in place fails; the successor retains a verifiable parent link. |
| A07 | Add machine-readable research-state labels (`PLANNED`, `RUNNING`, `COMPLETE_UNVERIFIED`, `VERIFIED_SUPPORTED`, `VERIFIED_NOT_SUPPORTED`, `INCONCLUSIVE`, `BLOCKED`). | State transitions are validated and an incomplete evidence bundle cannot transition to a supported state. |
| A08 | Introduce a formal `EvidenceBundle` manifest that includes raw logs, stdout/stderr, configs, checkpoints, metrics, predictions, environment lock, input hashes, and verifier reports. | The bundle verifier detects a missing, substituted, or unlisted file and reports the owning experiment. |

## B. Training and evaluator capability

| ID | Deliverable | Acceptance evidence |
|---|---|---|
| B01 | Build a generic local executor that consumes an Atlas experiment graph and runs commands in dependency order, emitting the same artifact contract used by Percy. | A five-node fixture executes locally, preserves task statuses, and produces a verification-ready bundle. |
| B02 | Provide a declarative training-job contract for Python entrypoints, arguments, environment variables, resource limits, checkpoint paths, and expected metrics. | A schema-validated job runs a reference trainer; an undeclared output is rejected. |
| B03 | Add resumable checkpoint recovery with atomic checkpoint writes, last-known-good metadata, RNG state, optimizer state, and data-cursor state. | Interrupting a reference training job and resuming it reproduces the uninterrupted run’s next-step metrics within protocol tolerance. |
| B04 | Add deterministic seed fan-out with language/framework seed capture and per-worker seed derivation. | Repeated CPU reference runs produce identical predictions and a manifest lists every derived seed. |
| B05 | Implement evaluator plugins for classification, regression, calibration, ranking, segmentation, sequence prediction, and generative likelihood, each with typed metric schemas. | Plugin fixtures produce metric artifacts that validate against their declared schema and reject invalid units/directions. |
| B06 | Implement paired treatment-control evaluation using matched seed, split, initialization, and budget identifiers. | The evaluator rejects an unpaired comparison and reports paired effect estimates for a valid fixture. |
| B07 | Add bootstrap confidence intervals and paired permutation tests with protocol-declared resampling count and random seed. | A synthetic known-effect study returns the expected interval/test decision and saves resample provenance. |
| B08 | Add multiplicity handling for declared secondary endpoints and ablation families (Holm or Benjamini–Hochberg, selected in protocol). | A multi-endpoint fixture saves raw and adjusted values and blocks unsupported claims based only on unadjusted significance. |
| B09 | Implement a baseline registry with immutable source revision, command, budget, preprocessing, metric adapter, and cited origin. | A run export refuses a baseline whose revision or budget is absent; the report prints comparator provenance. |
| B10 | Add budget-normalized comparison runners that record wall time, accelerator hours, peak memory, parameters, FLOPs estimate, and energy when available. | A reference comparison produces a per-arm cost table alongside accuracy metrics. |
| B11 | Add fixed validation selection followed by locked final retraining and one-time evaluation workflows. | A fixture demonstrates that selection artifacts cannot read a locked evaluation split before finalization. |

## C. Domain adapters and scientifically meaningful extensions

| ID | Deliverable | Acceptance evidence |
|---|---|---|
| C01 | Finish the LAM-JEPA ARC-v3 adapter as a full study adapter: parse producer outputs into normalized predictions, metrics, resource records, and verifier artifacts. | A fixture from the frozen adapter produces a valid Atlas evidence bundle without touching ARC test data. |
| C02 | Implement the Weather-JEPA frozen-protocol adapter with explicit DKE/ASI identifiers, source and dataset hashes, and the branch’s existing pre-outcome authorization gates. | Dry-run emits the frozen plan; execution refuses semantic protocol drift and prohibited evaluation identifiers. |
| C03 | Build a modality-agnostic adapter SDK for tabular, vision, language, temporal/spatiotemporal, and program-induction studies. | Minimal adapters from three modalities pass the same contract suite and emit interoperable bundles. |
| C04 | Add a dataset-card contract with provenance, license, consent/use limits, unit of analysis, label origin, known gaps, split procedure, and prohibited fields. | Execution blocks a dataset without a complete card or with a split hash mismatch. |
| C05 | Implement split constructors for IID, group, temporal, geographic, long-tail, and out-of-distribution settings. | Each constructor writes index files and diagnostic leakage checks; group/temporal overlap causes failure. |
| C06 | Add corruption and intervention suites: missingness, label noise, covariate shift, calibration shift, temporal drift, and adversarially hard but semantically valid examples. | A declared robustness matrix generates reproducible perturbation descriptors and condition-level metrics. |
| C07 | Add subgroup/fairness evaluation with protocol-declared groups, minimum support, uncertainty, and privacy-safe suppression policy. | A fixture outputs subgroup intervals, suppresses undersized groups, and refuses undeclared post-hoc slicing. |
| C08 | Implement out-of-distribution detection and selective-prediction adapters (AUROC, AUPR, risk-coverage, expected calibration error). | A controlled in/out-distribution fixture generates valid curves and a threshold policy artifact. |
| C09 | Add counterfactual and mechanism ablation study templates that require a matched compute budget and independently verify which component changed. | A component-off experiment cannot be exported until its only allowed implementation delta is declared and hashed. |
| C10 | Implement training-strategy variants—supervised, self-supervised pretraining, curriculum, multi-task, and transfer—under a common compute/data accounting interface. | A reference adapter executes two strategies and produces a budget-comparable report. |
| C11 | Add an intervention-aware causal-study template with treatment assignment, confounder assumptions, estimand, overlap diagnostics, and sensitivity analysis. | A synthetic causal dataset recovers a known effect and fails when positivity diagnostics are violated. |
| C12 | Add external-replication packets that allow a different machine or executor to rerun an evidence bundle using only pinned inputs. | A clean-environment replay reproduces the recorded fixture result and emits a distinct replication receipt. |

## D. Experiment interface and run operations

| ID | Deliverable | Acceptance evidence |
|---|---|---|
| D01 | Expand `atlas` into a complete study CLI: `init`, `validate`, `plan`, `run`, `resume`, `status`, `verify`, `classify`, `compare`, `report`, and `export`. | `atlas --help` exposes all commands; an end-to-end fixture uses no private Python API. |
| D02 | Add YAML/TOML protocol authoring with canonical JSON normalization and line-specific validation errors. | Equivalent YAML and JSON yield the same digest; malformed files identify the exact field path. |
| D03 | Implement `atlas run --dry-run` to render the experiment graph, resolved commands, inputs, expected artifacts, resource requests, and test-access policy without executing. | The generated plan is deterministic and includes every graph node and dependency. |
| D04 | Add a terminal run dashboard showing DAG state, completed/failed nodes, current command, elapsed time, resource use, artifact completeness, and stop reason. | An integration fixture streams state transitions and marks a failed node without hiding dependent blocked nodes. |
| D05 | Add a local read-only web dashboard for run browsing, protocol diffs, metric comparison, artifact download, and evidence-verification status. | Browser integration tests load a fixture study and never expose a command that can mutate an executed protocol. |
| D06 | Implement run comparison with compatibility checks for dataset/split digest, metric definition, baseline revision, compute budget, and protocol lineage. | Comparing incompatible runs gives a field-level refusal; compatible runs produce paired effect summaries. |
| D07 | Add run tags and structured query filters for scientific domain, adapter, protocol state, dataset, metric, baseline, seed, and parent protocol. | A fixture ledger query returns the expected subset and is stable across serialization/reload. |
| D08 | Add event-stream JSONL records for submission, start, heartbeat, checkpoint, artifact write, failure, retry, verification, and classification. | Replaying an event log rebuilds the dashboard state exactly and detects sequence gaps. |
| D09 | Provide export formats for a signed/archive-ready evidence package, CSV metric tables, Parquet prediction tables, and static HTML report. | Exports contain matching run/protocol digests and a verifier detects cross-format drift. |
| D10 | Add executor backends for local subprocess, Percy, and a documented pluggable remote adapter, all mapped to one canonical terminal-status vocabulary. | The same fixture graph validates identically after local and simulated-Percy execution. |

## E. Model, error, and evidence inspection

| ID | Deliverable | Acceptance evidence |
|---|---|---|
| E01 | Create prediction-table and residual-table schemas with example ID, split, target, prediction, uncertainty, model version, baseline prediction, and condition ID. | A metric report can be recomputed from the retained table and rejects rows from the wrong split. |
| E02 | Generate task-appropriate error atlases: confusion matrices and reliability diagrams for classification; residual, calibration, and quantile plots for regression. | A fixture produces deterministic PNG/SVG figures and records the data/query digest behind each panel. |
| E03 | Add paired per-example treatment-versus-control delta analysis with stratification by declared condition and subgroup. | The report lists wins/losses/ties and paired uncertainty, with no undeclared subgroup in the output. |
| E04 | Implement representation inspection adapters for embedding projections, nearest-neighbor retrieval, cluster purity, collapse diagnostics, and temporal trajectories. | A toy encoder emits a retained embedding artifact and all plots link to its checksum. |
| E05 | Add attention/memory/state inspection contracts for models that expose attention maps, latent predictions, memory slots, or recurrent state. | A model adapter saves a bounded, privacy-safe probe set and the viewer links each visualization to its input examples. |
| E06 | Implement intervention and attribution reports with method/version, baseline input, target output, stability checks, and caveats. | A fixture includes an attribution stability score; the report is blocked if the method is not declared in the protocol. |
| E07 | Add data-quality dashboards for duplicates, near-duplicates, train/evaluation overlap, missingness, label disagreement, group leakage, and distribution shift. | A contaminated fixture reports the injected defects and marks affected comparisons inadmissible. |
| E08 | Add failure forensics that join stack traces, environment changes, input hashes, resource traces, and last checkpoint into a structured incident report. | A simulated out-of-memory run creates a reproducible incident bundle without overwriting its prior evidence. |

## F. Reproducibility, integrity, and lifecycle

| ID | Deliverable | Acceptance evidence |
|---|---|---|
| F01 | Implement one-command, dependency-pinned reference studies for the Atlas core fixture, ARC adapter dry-run, and Weather-JEPA dry-run. | Fresh isolated environments reproduce all planning/verification outputs byte-for-byte where timestamps are excluded. |
| F02 | Add environment capture for Python/OS, package lock, hardware/driver details, container digest, Git revision, uncommitted-change policy, and locale/timezone. | The verifier flags a missing or inconsistent environment record. |
| F03 | Add hash-linked provenance from final claim report through metric table, prediction table, checkpoint, config, dataset/split, and source revision. | A provenance traversal command reaches all roots and detects a substituted intermediate artifact. |
| F04 | Implement deterministic artifact retention policies with explicit “required,” “optional,” and “redactable” classes; record every deletion/redaction as an event. | Policy tests reject deletion of required evidence and retain a redaction receipt for allowed removals. |
| F05 | Add compatibility/version migration tests for every public manifest, bundle, and Percy schema. | Older fixture schemas either migrate with a recorded transform or fail with a precise unsupported-version error. |
| F06 | Implement a research ledger database with append-only run and evidence records, content digests, parent-child lineage, and queryable state. | Attempts to alter a finalized record are rejected; a ledger export verifies against stored digests. |
| F07 | Add resource admission controls—declared CPU/GPU/memory/storage budget, quota checks, and no-rising-pageout guard for bounded numerical jobs. | A job above its budget is refused before execution with a retained refusal record. |

## G. Paper-ready evidence outputs

| ID | Deliverable | Acceptance evidence |
|---|---|---|
| G01 | Build a results-table generator for primary endpoints, baseline deltas, confidence intervals, p-values/adjustments, seeds, sample counts, and compute cost. | A generated LaTeX/CSV/Markdown table agrees numerically with the frozen metric artifacts. |
| G02 | Generate a protocol-derived experiment table listing every condition, control, seed count, split, intervention, budget, and terminal status. | The table is complete for a fixture manifest and cannot omit failed or blocked conditions. |
| G03 | Create publication figures for main performance, robustness curves, calibration, compute-performance trade-off, and per-condition effects. | Each figure has a machine-readable provenance sidecar and regenerates from the evidence bundle. |
| G04 | Add an ablation-report generator that verifies matched controls and reports effect size/uncertainty for every declared architectural component. | A missing mechanism-off control produces a visible “not evaluable” row rather than a fabricated result. |
| G05 | Produce a statistical-analysis report that states estimand, unit, test, assumptions, resampling seed, multiplicity treatment, exclusions, and all terminal outcomes. | A synthetic fixture outputs the exact precomputed analysis report; an undeclared exclusion fails validation. |
| G06 | Generate a limitations and validity-threats section from machine-readable protocol fields: internal validity, construct validity, external validity, compute limits, data limits, and negative findings. | The report contains concrete protocol-linked limitations and refuses unsupported causal/generalization language. |
| G07 | Add claim-evidence matrices mapping every manuscript claim to protocol IDs, result artifacts, evidence completeness, classification, and permitted wording. | A claim without complete backing is rendered as unsupported/inconclusive and cannot appear as a positive conclusion. |
| G08 | Create a reproducibility appendix generator with commands, expected checksums, environment, licenses, resource footprint, data-access steps, and replay outcome. | A clean replay receipt is linked in the appendix, or the appendix truthfully records that replay has not occurred. |

## H. Public demonstration and research handoff

| ID | Deliverable | Acceptance evidence |
|---|---|---|
| H01 | Build a safe interactive “Protocol-to-Plan” demo that accepts a research question and displays the generated hypothesis, endpoints, experiment graph, artifact contract, and evidence gates. | The demo uses sample/non-outcome data, validates input, and never implies that a generated plan is a confirmed result. |
| H02 | Provide an executable walkthrough notebook that constructs a toy protocol, runs a deterministic local reference experiment, verifies evidence, classifies a predeclared claim, and visualizes errors. | `jupyter execute` completes from a fresh environment and leaves an evidence bundle matching the notebook report. |
| H03 | Publish an inference-style `atlas inspect-bundle` command that lets users explore a completed bundle, provenance graph, figures, predictions, and claim boundary without rerunning study code. | It opens a fixture bundle read-only and detects tampering before rendering contents. |
| H04 | Create a gallery of three domain adapters—ARC/program induction, weather/spatiotemporal prediction, and a small tabular/regression reference—with their scientific boundary labels visible. | Each card links to a protocol, dry-run or verified result state, and retained artifacts; no pre-outcome adapter is labeled validated. |
| H05 | Add a downloadable minimal reference bundle and tutorial for external developers to implement an adapter and pass the Atlas contract suite. | A separate clean checkout completes the tutorial adapter and produces a contract-conformant bundle. |
| H06 | Provide a static evidence explorer export suitable for archival hosting, including protocol, ledger, hashes, status timeline, figures, and claim report. | The exported site renders offline, links only retained files, and its index checksum is verified by `atlas inspect-bundle`. |

## Completion order

Start with A01–A08, B01–B11, and F01–F07: they establish the executable,
evidence-safe substrate. Then implement one real adapter end-to-end (C01) and
its inspection/reporting path (E01–E08 and G01–G08) before broadening to
Weather-JEPA or additional modalities. Public-facing work in H01–H06 should
only expose recorded status and evidence; it must preserve `INCONCLUSIVE`,
`BLOCKED`, and negative outcomes.
