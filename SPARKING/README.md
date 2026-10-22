# SPARKING

SPARKING is an evidence-bound study of support-aware composition-only materials property prediction. Its v3 successor tests whether target-independent inverse-frequency weighting improves randomized forests under declared leave-one-element-out band-gap shifts.

## Status

**SPARKING v3 is complete with disposition `DEVELOPMENT_NOT_SUPPORTED`.** The private Kaggle version-1 run completed all frozen O/S/In/Ge/Sb/Ba/Co/F development holdouts on dual Tesla T4 GPUs. The selected `extra_trees_alpha0.50_mean` model had macro MAE 0.709135 eV, versus 0.718258 eV for its aggregation-matched unweighted Extra Trees control. The 0.009123 eV improvement missed the preregistered strict threshold of greater than 0.010 eV by 0.000877 eV.

The selected model improved five of eight elements and improved by 0.041814 eV over the CatBoost ensemble, so those two admission conditions passed. The frozen gate was conjunctive, making the failed matched-control threshold decisive. Li, Zn, Ga, Se, and Te remain unopened, and no confirmatory or official-fold run is authorized.

The exact output archive has SHA-256 `8675064282ad9f74e54a4a7489665c992101c6b67b4e9de281ad46271895d049`. Source commit `5b57c71ae03174b69cde426676fa2000d2724dce` and source archive SHA-256 `9467b930ff71ce462aaaa9e3eece40fc0b61e8b45b1d0812f02ff994d72c0553` verified with no changed, extra, or missing source files. The run manifest verified, all eight raw prediction arrays were retained, and the independent verifier reproduced 320 recorded metrics. The hosted test suite passed 21/21.

See the [v3 results report](docs/DEVELOPMENT_RESULTS_V3.md), [frozen v3 protocol](docs/PROTOCOL_V3.md), [canonical research state](RESEARCH_STATE.json), and [independent verification receipt](artifacts/development-v3-verification.json).

SPARKING v2 remains preserved with its own verified negative development result. V3 is a separately versioned successor and does not rewrite the v1 or v2 evidence.

## Reproduce the verified v3 result

```bash
.venv/bin/python scripts/verify_successor_v3.py runs/kaggle-development-v3-r1 --protocol protocols/development-v3.json --output /tmp/development-v3-verification.json
.venv/bin/python scripts/analyze_successor_v3.py runs/kaggle-development-v3-r1 --output /tmp/development-v3-posthoc-analysis.json
```

The exact private hosted archive, checksum, environment freeze, source verification, and kernel log are retained in `artifacts/kaggle-development-v3-r1-output/`. The manifest-bound raw predictions and diagnostics are retained in `runs/kaggle-development-v3-r1/`.

## Claim boundary

The verified v3 result supports a negative development conclusion about one frozen frequency-weighting mechanism on one declared eight-element panel. It does not establish crystal structure, thermodynamic stability, synthesizability, safety, novelty, device performance, experimental discovery, protected-element generalization, or official Matbench-fold preservation. The post-hoc bootstrap is descriptive and does not override the preregistered point-estimate gate.
