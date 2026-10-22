# SPARKING v3 development result

## Disposition

`DEVELOPMENT_NOT_SUPPORTED`

The frozen v3 development run completed successfully, but its conjunctive admission gate failed. The selected model improved macro MAE by 0.009123345454 eV over the matched unweighted forest. The protocol required a strict improvement greater than 0.010000 eV, leaving a margin of -0.000876654546 eV relative to the threshold. No confirmatory run was launched.

## Identity and verification

- Frozen source commit: `5b57c71ae03174b69cde426676fa2000d2724dce`
- Source archive SHA-256: `9467b930ff71ce462aaaa9e3eece40fc0b61e8b45b1d0812f02ff994d72c0553`
- Private source dataset: `ryangomez010/sparking-source-v3-20261022`
- Private kernel version 1: `ryangomez010/sparking-v3-frequency-balanced-forest`
- Run identity: `kaggle-development-v3-r1`
- Exact output archive SHA-256: `8675064282ad9f74e54a4a7489665c992101c6b67b4e9de281ad46271895d049`
- Deterministic 25-file run-tree digest: `db9d6958bdac20f61ff118436c3b226fbe386584db8e8ac2fa97efbb573bd64c`
- Hosted tests: 21 passed
- Independent raw-array metric comparisons: 320
- Runtime: 3090.193 seconds
- Peak RSS: 1161.090 MiB
- Pageout delta: 0
- Hardware: two NVIDIA Tesla T4 GPUs, 15360 MiB each

Source verification reported no changed, extra, or missing files. The artifact manifest reported no changed, extra, or missing run artifacts. The downloaded archive checksum matched its declared checksum, and its extracted 25-file run tree matched the directly retained output byte-for-byte. Both the hosted and independent local verifiers returned `VERIFIED` with the same selected model and disposition.

## Frozen decision rule

The selected nonzero-frequency candidate had to satisfy every condition below.

| Condition | Observed | Required | Result |
|---|---:|---:|---|
| Macro MAE improvement over aggregation-matched unweighted Extra Trees | 0.009123345454 eV | > 0.010000 eV | **FAIL** |
| Improved elements over matched unweighted Extra Trees | 5/8 | >= 5/8 | pass |
| Macro MAE improvement over CatBoost ensemble | 0.041814193968 eV | > 0.020000 eV | pass |
| Complete finite manifest-valid evidence | yes | required | pass |
| Nondegenerate positive normalized weights | yes | required | pass |

## Model results

The frozen selection rule chose `extra_trees_alpha0.50_mean` at macro MAE 0.709134559411 eV. All candidates and controls are retained, including the failed ridge control.

| Model | Macro MAE (eV) |
|---|---:|
| `extra_trees_alpha0.50_mean` | 0.709134559 |
| `extra_trees_alpha0.50_median` | 0.709794728 |
| `extra_trees_alpha1.00_mean` | 0.709958535 |
| `extra_trees_alpha1.00_median` | 0.710322732 |
| `extra_trees_alpha0.00_median` | 0.718237571 |
| `extra_trees_alpha0.00_mean` | 0.718257905 |
| `direct_hgb_squared` | 0.748100908 |
| `catboost_ensemble` | 0.750948753 |
| `catboost_single_seed` | 0.771896017 |
| `ridge` | 3.715507220 |

Ridge reached 22.631720 eV MAE on Co and 3.715507 eV macro MAE. This failure is preserved and included in the table rather than removed as an outlier.

## Element-level results

Positive improvement means the selected model reduced MAE relative to its matched unweighted control.

| Element | Selected | Unweighted | CatBoost | HGB | Selected improvement vs unweighted |
|---|---:|---:|---:|---:|---:|
| O | 1.493477 | 1.484127 | 1.503988 | 1.438000 | -0.009350 |
| S | 0.743441 | 0.758817 | 0.733356 | 0.773738 | +0.015376 |
| In | 0.325357 | 0.322222 | 0.336932 | 0.339290 | -0.003135 |
| Ge | 0.316374 | 0.320560 | 0.356968 | 0.348471 | +0.004186 |
| Sb | 0.461581 | 0.465903 | 0.462735 | 0.475707 | +0.004322 |
| Ba | 0.645492 | 0.654891 | 0.652310 | 0.638589 | +0.009399 |
| Co | 0.149327 | 0.148855 | 0.165873 | 0.152619 | -0.000472 |
| F | 1.538027 | 1.590689 | 1.795428 | 1.818394 | +0.052662 |

Frequency balancing helped S, Ge, Sb, Ba, and F, and worsened O, In, and Co. The largest improvement was F at +0.052662 eV; the largest worsening was O at -0.009350 eV.

## Post-hoc uncertainty analysis

A deterministic 20,000-replicate bootstrap was run after the primary decision. It clusters by unique Matbench source row across overlapping element holdouts. This analysis was not preregistered and cannot override the frozen point-estimate gate.

- Selected versus matched unweighted: cluster 95% interval [0.006650256720, 0.011586486913] eV; probability improvement > 0 was 1.00000; probability improvement > 0.01 eV was 0.24445.
- Selected versus CatBoost: cluster 95% interval [0.026738211544, 0.056624417617] eV; probability improvement > 0.02 eV was 0.99805.

The interval straddling the 0.010 eV threshold shows the estimate is close to the admission boundary, but the preregistered rule used the observed point estimate and remains failed.

## Protected evaluation and claim boundary

The run records `protected_confirmatory_outcomes_observed=false` and `official_fold_outcomes_observed=false`. Li, Zn, Ga, Se, and Te remain unopened. Because development admission failed, the confirmatory protocol was not frozen and no confirmatory or official-fold execution was launched.

This result is limited to one frequency-weighting mechanism, one dataset, and the declared development panel. It is not evidence of synthesizability, stability, experimental discovery, or protected-element generalization.

## Reproduction

```bash
.venv/bin/python scripts/verify_successor_v3.py runs/kaggle-development-v3-r1 --protocol protocols/development-v3.json --output /tmp/development-v3-verification.json
.venv/bin/python scripts/analyze_successor_v3.py runs/kaggle-development-v3-r1 --output /tmp/development-v3-posthoc-analysis.json
```
