# Uncertainty-aware composition models for retrospective experimental band-gap candidate ranking

## Abstract

We evaluated a narrow materials-informatics workflow for predicting experimental inorganic band gaps from composition and ranking a fixed historical candidate pool. On 4,604 Matbench v0.1 records using the official five-fold outer splits, a three-seed ExtraTrees ensemble achieved MAE 0.481 eV and RMSE 0.906 eV; a regularized linear composition baseline achieved MAE 0.813 eV. Group-isolated split-conformal calibration produced 0.889 coverage for nominal 90% intervals. At a 50-label retrospective budget, calibrated target-window probability selected 38.0 compositions in a prespecified 1.1–1.6 eV band, compared with 3.6 for random selection. Element-holdout errors were materially worse than random-split performance for some elements. These results support software and protocol feasibility only: the ranking uses historical labels for post hoc evaluation and does not verify stability, synthesizability, or device performance.

## 1. Introduction

Materials-property benchmarks can test prediction and ranking machinery, but high benchmark utility is not itself materials discovery. We therefore scoped this work to one property, one public benchmark, and one intended decision: prioritize compositions likely to fall inside a prespecified experimental band-gap interval. The study separates official benchmark evaluation, custom element-holdout extrapolation, uncertainty calibration, and retrospective selection utility.

## 2. Data and methods

### 2.1 Data contract

The input was the composition-only `matbench_expt_gap` task (Matbench v0.1), sourced from Zhuo, Mansouri Tehrani, and Brgoch and curated by Dunn et al. The target is experimental band gap in eV. The pinned 4,604-row artifact has SHA-256 `783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2`. Exact official five-fold IDs came from the Matbench validation bundle with SHA-256 `94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d`.

An integrity audit found three canonical duplicates missed by string-level upstream curation. Exact official rows were retained solely for benchmark comparability. Calibration partitions and custom splits kept canonical groups intact, while the candidate pool collapsed them to unique compositions and retained all source IDs and target spans.

### 2.2 Models and uncertainty

Features were element fractions plus scale-invariant stoichiometric and atomic-number summaries. The baseline was standardized ridge regression with inner alpha selection. The proposed predictor averaged three fixed-seed ExtraTrees models. A label-blind 20% subset of each outer-training fold calibrated tree dispersion; normalized split-conformal scores produced prediction intervals. No structure, relaxed geometry, calculated property, or post-measurement metadata was used.

### 2.3 Endpoints

Prediction endpoints were MAE, RMSE, R², and Spearman correlation. Calibration endpoints were empirical coverage and interval width. Abstention was evaluated by retaining progressively lower-uncertainty subsets. Candidate success was an experimental gap in [1.1, 1.6] eV at budgets 10, 25, 50, and 100. Policies were random, greedy proximity to the target-window center, maximum uncertainty, and calibrated probability of the target interval.

## 3. Results

| Model | MAE (eV) | RMSE (eV) | R² | Spearman |
|---|---:|---:|---:|---:|
| Calibrated ExtraTrees ensemble | 0.481 | 0.906 | 0.607 | 0.772 |
| Ridge composition baseline | 0.813 | 1.161 | 0.354 | 0.647 |
| Train-median predictor | 0.976 | 1.744 | -0.456 | 0.000 |

Nominal 90% intervals achieved 0.889 coverage with width 1.604 eV. At budget 50, proposed, greedy, uncertainty, and random policies achieved precisions 0.760, 0.300, 0.000, and 0.072, respectively.

| Held-out element | Train n | Test n | Proposed MAE (eV) | Ridge MAE (eV) | 90% coverage |
|---|---:|---:|---:|---:|---:|
| Li | 4258 | 346 | 1.104 | 1.429 | 0.919 |
| Zn | 4391 | 213 | 0.499 | 0.763 | 0.817 |
| Ga | 4233 | 371 | 0.605 | 0.651 | 0.779 |
| Se | 3928 | 676 | 0.974 | 0.879 | 0.525 |
| Te | 4186 | 418 | 0.678 | 0.702 | 0.679 |

## 4. Discussion

The experiment demonstrates a reproducible route from a pinned composition dataset to predictions, calibrated intervals, abstention consequences, and a provenance-rich candidate ledger. It also exposes limits hidden by a headline error: canonical duplicates cross official folds, label conflicts remain, and element holdout can be much harder than randomized evaluation. The candidate policy is evaluated retrospectively and cannot establish a new material, thermodynamic stability, synthesis feasibility, or application performance.

## 5. Reproducibility and status

Source, split, configuration, predictions, checkpoints, failures, environment, and artifact hashes are retained under `runs/kaggle-development-v1-r1`. The result is **EXECUTED_DEVELOPMENT** and **NOT PROSPECTIVELY OR INDEPENDENTLY VALIDATED**.

## References

1. Dunn A, Wang Q, Ganose A, Dopp D, Jain A. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. *npj Computational Materials* 6, 138 (2020). https://doi.org/10.1038/s41524-020-00406-3
2. Zhuo Y, Mansouri Tehrani A, Brgoch J. Predicting the Band Gaps of Inorganic Solids by Machine Learning. *J. Phys. Chem. Lett.* 9, 1668–1673 (2018). https://doi.org/10.1021/acs.jpclett.8b00124
