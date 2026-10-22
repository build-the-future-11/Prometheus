# Results

Status: **EXECUTED_DEVELOPMENT — NOT PROSPECTIVELY OR INDEPENDENTLY VALIDATED**

## Completed scope

The repository was empty at intake, so the approved fallback objective was used: calibrated experimental band-gap prediction from composition and an explicitly evaluated uncertainty-aware policy over a fixed candidate pool. The only property task is `matbench_expt_gap`; no additional property projects were invented.

The executed study used all 4,604 Matbench v0.1 rows, official five-fold outer splits, three frozen model seeds, group-isolated calibration subsets, five separately labeled element-holdout studies, and a unique-composition candidate pool of 4601 candidates. The target is experimental band gap in eV.

## Official Matbench-split prediction results

| Model | MAE (eV) | RMSE (eV) | R² | Spearman |
|---|---:|---:|---:|---:|
| Calibrated ExtraTrees ensemble | 0.481 | 0.906 | 0.607 | 0.772 |
| Ridge composition baseline | 0.813 | 1.161 | 0.354 | 0.647 |
| Train-median predictor | 0.976 | 1.744 | -0.456 | 0.000 |

These are out-of-fold results on the exact official split IDs. They are not leaderboard submissions. Because equivalent canonical compositions cross some official train/test boundaries, see the data-integrity caveat below.

## Uncertainty and abstention

The 90% normalized split-conformal interval achieved 0.889 observed coverage with mean width 1.604 eV. Calibration rows for 50%, 80%, 90%, and 95% intervals are retained in `tables/calibration.csv`.

Uncertainty was operationally tested through abstention, not merely plotted. The lowest-uncertainty 20% had MAE 0.025 eV versus 0.481 eV without abstention. Rejected-set error and all retained fractions are recorded in `tables/abstention.csv`.

## Fixed-pool candidate-selection utility

Success was frozen before execution as an evaluation label in the 1.1–1.6 eV window. `proposed_calibrated_probability` ranks by a calibrated Gaussian working probability; it never receives evaluation labels. Random selection reports 64 deterministic replicates.

| Policy | Budget | Hits | Precision | Enrichment vs pool |
|---|---:|---:|---:|---:|
| greedy_predicted_property | 10 | 2.00 | 0.200 | 2.99× |
| proposed_calibrated_probability | 10 | 9.00 | 0.900 | 13.44× |
| random | 10 | 0.77 | 0.077 | 1.14× |
| uncertainty | 10 | 0.00 | 0.000 | 0.00× |
| greedy_predicted_property | 25 | 6.00 | 0.240 | 3.59× |
| proposed_calibrated_probability | 25 | 22.00 | 0.880 | 13.15× |
| random | 25 | 1.75 | 0.070 | 1.05× |
| uncertainty | 25 | 0.00 | 0.000 | 0.00× |
| greedy_predicted_property | 50 | 15.00 | 0.300 | 4.48× |
| proposed_calibrated_probability | 50 | 38.00 | 0.760 | 11.35× |
| random | 50 | 3.58 | 0.072 | 1.07× |
| uncertainty | 50 | 0.00 | 0.000 | 0.00× |
| greedy_predicted_property | 100 | 34.00 | 0.340 | 5.08× |
| proposed_calibrated_probability | 100 | 63.00 | 0.630 | 9.41× |
| random | 100 | 7.03 | 0.070 | 1.05× |
| uncertainty | 100 | 0.00 | 0.000 | 0.00× |

The candidate ledger is retrospective. A ranked formula is not evidence of stability, synthesizability, safety, device utility, or independent verification.

## Separately labeled element-holdout extrapolation

These are custom composition-family studies, not official Matbench leaderboard results.

| Held-out element | Train n | Test n | Proposed MAE (eV) | Ridge MAE (eV) | 90% coverage |
|---|---:|---:|---:|---:|---:|
| Li | 4258 | 346 | 1.104 | 1.429 | 0.919 |
| Zn | 4391 | 213 | 0.499 | 0.763 | 0.817 |
| Ga | 4233 | 371 | 0.605 | 0.651 | 0.779 |
| Se | 3928 | 676 | 0.974 | 0.879 | 0.525 |
| Te | 4186 | 418 | 0.678 | 0.702 | 0.679 |

## Sanity checks

- Train-median prediction was evaluated in every official fold (table above).
- Three label-permutation controls produced mean MAE 1.119 eV versus 0.484 eV for the unpermuted fold-0 model. Pass: `True`.
- Every serialized fold checkpoint was immediately reloaded and required numerically identical predictions. Pass: `True`.
- Unit conversion, formula parsing, composition invariance, missing-data refusal, feature availability, split identity, group isolation, and ranking label separation are covered by the test suite.

## Negative findings and limitations

- The proposed ensemble improved MAE over ridge by only 0.332 eV; this is a development result, not external validation.
- The ranking gain is retrospective on measured Matbench labels and does not demonstrate prospective discovery.
- Element holdout degraded most for Li (MAE 1.104 eV).
- Three canonical composition groups remain duplicated in the upstream artifact; two cross official train/test boundaries in four folds. Official metrics are retained for comparability but are not composition-leakage-free.
- Structures and polymorph identity are absent. No structure claim can be made, and no neural structure model was justified.
- The Matbench metadata says compositions were deduplicated and repeats spanning more than 0.1 eV removed, but the pinned artifact contains three canonical duplicate groups with spans 0.58–2.30 eV. All six source rows are preserved in `data/manifests/canonical_duplicate_audit.json`.
- Candidate evaluation reuses known experimental labels only after ranking; there is no new calculation, synthesis, or wet-lab confirmation.

## Execution receipt

- Run ID: `kaggle-development-v1-r1`
- Started UTC: `2026-10-20T17:35:59.483128Z`
- Finished UTC: `2026-10-20T17:36:59.170661Z`
- Duration: `59.7` seconds
- Python: `3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0]`
- Dataset SHA-256: `783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2`
- Official split SHA-256: `94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d`
- Peak process RSS: `387.6` MiB (limit 512.0 MiB)
- Pageout delta: `0`
- Tests: see `test_receipt.json`; artifact hashes: `artifact_manifest.json`

## Claim boundary

Implemented, numerically executed, and locally tested are distinct from scientifically validated, prospectively discovered, release-certified, or independently reproduced. This run establishes only an internally reproducible development result on a public historical benchmark.
