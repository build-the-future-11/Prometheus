# Reproduction and continuation

Run all commands from `/Volumes/PRO-BLADE/Atlas/NEUROGENESIS`. The workflow has no third-party runtime dependencies.

## Validate the code and protocol

```bash
rtk python3 -m compileall -q neurogenesis tests
rtk python3 -m unittest discover -s tests -v
rtk python3 -m neurogenesis validate \
  --protocol configs/protocol.json \
  --config configs/development.json
```

## Verify the retained run without overwriting its receipt

```bash
rtk python3 scripts/verify_bundle.py \
  --root . \
  --bundle results/development/shared-baselines-v1-20261019 \
  --output /tmp/neurogenesis-verification.json
```

The verifier checks current protocol, config, and producer-source hashes; every receipt/result and result/trace hash; split and method admission; trace lengths and timing; finite values; checkpoints; resource ceilings; and the zero protected-evaluation count.

## Regenerate the verified development analysis and paper package

Use fresh output paths; neither generator overwrites a prior analysis or package.

```bash
rtk python3 scripts/analyze_bundle.py \
  --bundle results/development/shared-baselines-v1-20261019 \
  --output analysis/development-baselines-v1-PUBLICATION_ID

rtk python3 scripts/verify_analysis.py \
  --root . \
  --bundle results/development/shared-baselines-v1-20261019 \
  --analysis analysis/development-baselines-v1-PUBLICATION_ID \
  --output analysis_receipts/development-baselines-v1-PUBLICATION_ID.json

rtk python3 scripts/assemble_submission.py \
  --root . \
  --analysis analysis/development-baselines-v1-PUBLICATION_ID \
  --output submission/neurogenesis-v1-development-baselines-PUBLICATION_ID

cd submission/neurogenesis-v1-development-baselines-PUBLICATION_ID/paper
rtk tectonic -X compile --outdir /tmp/neurogenesis-paper-PUBLICATION_ID paper.tex
```

`analyze_bundle.py` is read-only with respect to the input bundle and generates JSON, CSV, LaTeX, an offline HTML report, all-attempt and policy-mean rolling-MSE SVGs, and a stability--plasticity SVG. `verify_analysis.py` regenerates each derived artifact byte-for-byte from the retained raw record.

## Repeat the admitted development matrix

Use a fresh output path. The runner refuses to overwrite any prior attempt bundle.

```bash
rtk python3 -m neurogenesis run-development \
  --protocol configs/protocol.json \
  --config configs/development.json \
  --output results/development/shared-baselines-v1-REPLICATION_ID

rtk python3 -m neurogenesis aggregate \
  --bundle results/development/shared-baselines-v1-REPLICATION_ID \
  --output results/development/shared-baselines-v1-REPLICATION_ID/aggregate

rtk python3 scripts/verify_bundle.py \
  --root . \
  --bundle results/development/shared-baselines-v1-REPLICATION_ID \
  --output results/development/shared-baselines-v1-REPLICATION_ID/verification.json
```

Do not run `configs/evaluation.json`. Protocol v1 has no evaluation entry point and rejects the evaluation split.

## Exact continuation after an owner scope decision

The next action is documentary before it is computational:

1. Add a dated owner decision under `decisions/` selecting exactly `learned_rule` or `memory_weight`.
2. Append `SCOPE_HISTORY.md`; do not edit away the unresolved record.
3. Create `configs/protocol_v2.json` and mechanism-specific development/evaluation manifests with total cost and falsifier thresholds frozen before outcomes.
4. Implement only the selected interface from `neurogenesis/interfaces.py` and its required controls.
5. Run CPU correctness tests, then a new versioned development bundle. Keep evaluation blocked until those receipts pass and owner admission is explicit.

Suggested validation command after those files exist:

```bash
rtk python3 -m unittest discover -s tests -v
rtk python3 -m neurogenesis validate \
  --protocol configs/protocol_v2.json \
  --config configs/development_v2.json
```

There is intentionally no continuation command for protected evaluation under protocol v1.

