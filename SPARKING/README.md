# SPARKING

SPARKING is an evidence-bound study of support-aware composition-only materials
property prediction. The model is a regularized linear extrapolator plus a
nonlinear residual learner. The residual is attenuated only when the query lies
outside the training support measured by nearest-neighbour distance in a fixed
composition feature space.

The scientific question is deliberately narrow: does this support gate improve
leave-one-element-out experimental-band-gap prediction relative to the same
residual model without the gate, while preserving official Matbench accuracy?

## Status

The predecessor study in `../ML4-Material Discovery` remains immutable and
resource-blocked. SPARKING is a separately versioned successor. Its current
machine-readable disposition is always recorded in `RESEARCH_STATE.json`.

## Reproduce

```bash
UV_CACHE_DIR=/tmp/sparking-uv-cache uv sync --extra dev
UV_CACHE_DIR=/tmp/sparking-uv-cache uv run sparking prepare
UV_CACHE_DIR=/tmp/sparking-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/sparking-uv-cache uv run sparking preflight
UV_CACHE_DIR=/tmp/sparking-uv-cache uv run sparking develop --run-id development-v1
```

The confirmatory command is disabled until a development receipt is explicitly
frozen into `protocols/confirmatory-v1.json`. No partial run is a scientific
result. `sparking verify` checks every retained file against its manifest.

## Claim boundary

This repository predicts one experimental property from composition only. It
does not establish crystal structure, thermodynamic stability, synthesizability,
safety, novelty, device performance, or experimental discovery.

