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

Development completed on Kaggle on 2026-10-21: all eight gate settings and four
baselines were evaluated across Cu/Ag/B, with the downloaded evidence verified.
The best numerical gate improved macro-MAE by only 0.000249 eV over the ungated
control. The frozen tie rule selected quantile 0.99, power 1. Protected
confirmation remains unrun. See [development results](docs/DEVELOPMENT_RESULTS_V1.md).

Follow-up [development diagnostics](docs/DEVELOPMENT_DIAGNOSIS_V1.md) reproduced
the original scores and identified a collapsed absolute-error direct boosting
baseline. With the same tree settings, squared-error direct boosting achieved
0.760592 eV macro-MAE versus 0.952610 for the original ungated residual model.
The selected gate affects only 1 of 824 development prediction cases.
Confirmation is on hold for development review; changes require a versioned
successor rather than edits to the frozen v1 study.

A separately versioned [v2 successor protocol](docs/PROTOCOL_V2.md) is now
frozen before execution. It replaces the ineffective distance gate with a
training-only pseudo-element risk calibrator around a five-seed GPU CatBoost
ensemble, adds corrected and strong baselines, and has an explicit development
stop rule. V1 evidence remains immutable. V2 has no numerical results yet and
does not authorize protected confirmation until its development admission rule
passes.

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

