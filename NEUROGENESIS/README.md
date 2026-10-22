## Current research outcome (2026-10-21)

**MIXED:** frozen synthetic confirmation supports 36.01% lower paired error and 58.55% fewer weight updates; real transfer is invalidated by six numerical failures. [Results, current status and next actions](RESULTS_CURRENT.md). [Final paper](successors/memory_v3/paper/main.pdf). Kaggle execution remains blocked.

# Current NEUROGENESIS research

Start with [RESEARCH_STATE.md](RESEARCH_STATE.md), the canonical live status.

The active successor is [memory v3](successors/memory_v3/README.md), with a fully implemented bounded RLS predictor bank, delayed-label scoring, strong baselines, frozen selection, raw traces, and reproducibility tooling. Historical v1 and v2 evidence is preserved. The branch name identifies the overall memory campaign; the scientific model version is v3.

The historical description below applies only to protocol v1. Its unresolved status and protected-evaluation lock are preserved and do not describe the separately authorized successor.

---

# NEUROGENESIS

NEUROGENESIS is an independent adaptive-learning research workspace. Its scientific mechanism is currently **unresolved**: no local dated owner decision establishes whether the project is about learned losses/update rules or bounded memory that substitutes for some parameter updates.

The current repository therefore implements only the explicitly permitted common layer:

- deterministic sequential task families with delayed labels, distractors, conflicting regimes, returning regimes, and a novel regime;
- prediction-before-label timing enforcement;
- fixed meta-training, development, and protected evaluation identities;
- no-adaptation, standard SGD, and mechanism-off standard Adam baselines;
- complete cost accounting, checkpoints, per-step traces, failed-attempt retention, aggregation, and independent verification;
- separate abstract interfaces for the two candidate scientific lines;
- a fail-closed gate that denies candidate or protected-evaluation execution while scope is unresolved.

No learned mechanism and no memory-weight mechanism has been implemented or evaluated. The retained results characterize development baselines only.

## Quick verification

```bash
rtk python3 -m unittest discover -s tests -v
rtk python3 -m neurogenesis validate \
  --protocol configs/protocol.json \
  --config configs/development.json
rtk python3 scripts/verify_bundle.py \
  --root . \
  --bundle results/development/shared-baselines-v1-20261019 \
  --output /tmp/neurogenesis-verification.json
```

The checked-in verification receipt is at `results/development/shared-baselines-v1-20261019/verification.json`. See `PROJECT.md`, `PROTOCOL.md`, `RESULTS.md`, and `REPRODUCE.md` for the scientific and evidentiary boundaries.

