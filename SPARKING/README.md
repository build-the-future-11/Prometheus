# SPARKING

SPARKING is an evidence-bound study of support-aware composition-only materials property prediction. It tests whether training-only estimates of direct-model harm can improve leave-one-element-out band-gap prediction by adaptively shrinking a nonlinear ensemble toward ridge regression.

## Status

**SPARKING v3 is frozen and Kaggle-run-ready; it has no scientific result yet.** V3 tests a target-independent frequency-balanced Extra Trees mechanism on a fresh O/S/In/Ge/Sb/Ba/Co/F development panel. The exact private source packager, dual-GPU launcher, production evaluator, raw-artifact writer, and independent verifier are implemented. The local suite passes 21/21 and the bounded CPU integration dry run passes. Li, Zn, Ga, Se, and Te remain protected.

The frozen contract and adversarial flaw audit are in [the v3 protocol](docs/PROTOCOL_V3.md). The private Kaggle kernel is configured as `ryangomez010/sparking-development-v3-frequency-balanced-forest-gpu`, backed by source dataset `ryangomez010/sparking-source-v3-20261022`. A hosted completion is required before any v3 performance claim.

**SPARKING v2 remains complete with disposition `DEVELOPMENT_NOT_SUPPORTED`.** A private Kaggle version-1 run completed the frozen Cu/Ag/B development protocol on dual Tesla T4 GPUs. All 180 CatBoost fits completed, the hosted test suite passed 18/18, and an independent local verifier recomputed 156 metrics from the downloaded raw arrays with a valid manifest.

The selected risk gate at temperature 0.40 eV had macro MAE 0.779667 eV versus 0.779536 eV for the identical ungated CatBoost ensemble, an improvement of -0.000130 eV. The frozen admission threshold was greater than +0.020 eV. Although Ag and B improved by less than 0.001 eV each, Cu worsened by 0.001921 eV. The protected Li, Zn, Ga, Se, and Te outcomes remain unopened, and v2 confirmation is not authorized.

The strongest development control was a three-seed Extra Trees ensemble at 0.728675 eV macro MAE. This is a secondary development finding, not a claim of state-of-the-art performance or a basis for opening protected outcomes.

See the [v2 results report](docs/DEVELOPMENT_RESULTS_V2.md), [frozen v2 protocol](docs/PROTOCOL_V2.md), [canonical research state](RESEARCH_STATE.json), and [independent verification receipt](artifacts/development-v2-verification.json).

SPARKING v1 remains preserved at commit `c04873557b522962833aafca3759caf6a3fa488b`, including its verified CPU development result and diagnostic failure analysis. V2 is an additive successor; it does not rewrite v1 evidence.

## Reproduce the verified result

```bash
.venv/bin/python scripts/verify_successor_v2.py runs/kaggle-development-v2-r1
MPLCONFIGDIR=/tmp/sparking-matplotlib .venv/bin/python scripts/analyze_successor_v2.py
```

The exact private hosted archive, checksum, environment freeze, source verification, and kernel log are retained in `artifacts/kaggle-development-v2-r1-output/`. The manifest-bound raw predictions and pseudo-calibration arrays are retained in `runs/kaggle-development-v2-r1/`.

## Claim boundary

This repository predicts one experimental property from composition only. The verified v2 result supports a negative development conclusion about one frozen gating mechanism on Cu/Ag/B. It does not establish crystal structure, thermodynamic stability, synthesizability, safety, novelty, device performance, experimental discovery, protected-element generalization, or official Matbench-fold preservation.
