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

