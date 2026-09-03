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

## Core design

- **Atlas = scientist/planner**: hypotheses, experiment matrices, stopping rules, evidence requirements.
- **Percy = executor**: implementation and compute tasks.
- **ProofForge = verifier**: claim-to-evidence validation and reproducibility checks.

Atlas never marks a claim as supported merely because a run completed. A claim must have explicit metrics, baselines, controls, and acceptance criteria.

## Current milestone

`v0.1`: deterministic research-run manifest generation with validation and JSON serialization.
