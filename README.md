# Atlas Research Engine

Atlas is an execution-oriented research orchestration layer for turning a research question into a structured, reproducible experiment plan and evidence package.

## What Atlas does

Atlas takes a research question and produces a concrete research run with:

1. a falsifiable hypothesis,
2. explicit success/failure criteria,
3. an experiment matrix,
4. reproducibility metadata,
5. executor-ready tasks,
6. verification hooks for downstream systems such as Percy and ProofForge.

The first milestone is intentionally local and dependency-light: a deterministic planner that can generate a rigorous run manifest without requiring external APIs.

## Quick start

```bash
python -m atlas.cli init-run \
  --question "Does latent prediction improve robustness under distribution shift?" \
  --output runs/robustness.json
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Atlas → Percy task export

Atlas can export any run manifest into a dependency-free JSON task bundle without importing or installing Percy:

```bash
python -m atlas.cli export-percy \
  --manifest runs/robustness.json \
  --output runs/robustness.percy.json \
  --working-directory /workspace/project \
  --command-template 'python run_experiment.py --experiment {experiment_id} --seed {seed} --condition {perturbation} --output {artifact_dir}'
```

The export uses schema `atlas.percy-tasks.v0.1`. Every experiment becomes one durable task with a deterministic task ID, working directory, command, timeout, retry policy, resource request, Atlas run/experiment metadata, and the experiment's required artifacts. The export also creates one parent verification task depending on every experiment task.

Available command-template placeholders are `{run_id}`, `{experiment_id}`, `{experiment_name}`, `{seed}`, `{perturbation}`, `{treatment}`, `{control}`, and `{artifact_dir}`.

### Minimal Percy ingestion contract

A Percy-side importer only needs to:

1. reject unknown `schema_version` values rather than guessing compatibility;
2. ingest every `tasks[]` entry using `task_id` as the stable external identity;
3. preserve `dependencies`, `working_directory`, `command`, `timeout_seconds`, `retry_policy`, `resources`, `artifact_directory`, `expected_artifacts`, and `metadata`;
4. execute the parent verification task only after every dependency reaches a terminal state;
5. write the `status_contract.path` JSON object mapping each experiment task ID to its terminal status, using `SUCCEEDED` for success;
6. never mark the parent verification task successful when a dependency failed/missing or a required artifact is absent.

The verification command embedded in the parent task runs Atlas itself and fails closed:

```bash
python -m atlas.cli verify-percy \
  --export runs/robustness.percy.json \
  --statuses runs/<run-id>-task-statuses.json \
  --artifact-root runs/<run-id>-artifacts
```

Artifacts are expected beneath `<artifact-root>/<atlas_experiment_id>/...` using each experiment's `required_artifacts` paths. Absolute paths and parent-directory traversal in required artifact declarations are rejected during verification.

## Frozen LAM-JEPA ARC-v3 adapter

Atlas includes an adapter for the checked-in `integrations/lam_jepa_arc_v3.json` reproduction contract. CI exercises only the dry-run path; it does not execute the outcome-bearing benchmark.

Inspect the frozen execution plan without running anything:

```bash
python -m atlas.lam_jepa_adapter \
  --integration integrations/lam_jepa_arc_v3.json \
  --dry-run
```

An actual run requires a clean local `vertex-studyAI/LAM-JEPA` checkout at the exact frozen SHA and Python 3.11:

```bash
python -m atlas.lam_jepa_adapter \
  --integration integrations/lam_jepa_arc_v3.json \
  --repo /path/to/LAM-JEPA \
  --output-root /path/to/fresh/atlas-evidence
```

The adapter validates the integration contract, refuses the ARC test parquet before and after execution, checks the exact source SHA, executes in an isolated copied workspace, gates the producer on successful protocol preflight, runs the independent verifier whenever producer output exists, records command exit codes/stdout/stderr, freezes the Python package list, hashes train/validation datasets, retains producer/verifier outputs, and writes one checksum manifest over the retained evidence. The scientific verdict remains `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`; this adapter does not authorize rescue tuning or test-split access.

## Core design

- **Atlas = scientist/planner**: hypotheses, experiment matrices, stopping rules, evidence requirements.
- **Percy = executor**: implementation and compute tasks.
- **ProofForge = verifier**: claim-to-evidence validation and reproducibility checks.

Atlas never marks a claim as supported merely because a run completed. A claim must have explicit metrics, baselines, controls, and acceptance criteria.

## Current milestone

`v0.1`: deterministic research-run manifest generation with validation and JSON serialization, a versioned Atlas → Percy durable-task export contract with fail-closed artifact verification, plus evidence-safe execution adapters and verification primitives.
