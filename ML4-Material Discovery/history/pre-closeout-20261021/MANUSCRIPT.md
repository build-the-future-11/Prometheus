# Uncertainty-aware composition models for experimental band-gap candidate ranking

Manuscript status: **METHODS DRAFT — RESULTS BLOCKED, NOT SUBMISSION-READY**

## Abstract

This draft specifies a reproducible evaluation of composition-only experimental band-gap prediction, calibrated uncertainty, abstention, and fixed-pool candidate ranking on Matbench v0.1. The software, data contract, official split mapping, baselines, proposed ensemble, calibration procedure, policy evaluator, and custom element-holdout studies are implemented and locally tested. However, the prespecified no-rising-pageout compute gate refused all four frozen development runs on the current macOS host. Across retained attempts, only two of five fold checkpoints were produced in one run and one fold in another; the latest run stopped at data validation. Consequently, no aggregate error, calibration, ranking, or extrapolation result is reported. The correct scientific status is `NOT_EVALUATED_RESOURCE_BLOCKED`.

## 1. Objective

The study asks one bounded question: can composition-only models estimate experimental inorganic band gap and improve retrospective selection of formulas whose measured gaps fall in a prespecified 1.1–1.6 eV window? It does not address structure, stability, synthesis, device performance, or universal materials modeling.

## 2. Data

The frozen input is the 4,604-row Matbench v0.1 `matbench_expt_gap` task, derived from Zhuo, Mansouri Tehrani, and Brgoch and curated for Matbench by Dunn et al. The target is experimental band gap in eV. Inputs contain only composition strings.

A local scale/order-invariant audit identified 4,601 canonical compositions and three duplicate pairs with target spans of 0.58–2.30 eV. Two duplicate groups cross official train/test boundaries in four folds. Exact official rows remain unchanged for benchmark comparability; local calibration and custom splits isolate canonical groups, and the candidate pool collapses each group while preserving every source value.

## 3. Methods

Formula-derived features comprise element fractions and scale-invariant stoichiometric and atomic-number summaries. Baselines are a training-median predictor and standardized ridge regression. The proposed predictor averages three ExtraTrees models with fixed seeds, 96 trees per seed, maximum depth 22, and one worker.

Within each official outer-training fold, a label-blind canonical-group split withholds approximately 20% for dispersion scaling and normalized split-conformal calibration. The final outer fold is untouched until evaluation. Prediction endpoints are MAE, RMSE, R², and Spearman correlation. Uncertainty endpoints are interval coverage/width and error after retaining successively lower-uncertainty subsets.

The candidate evaluator compares random selection, greedy predicted proximity to the window center, maximum uncertainty, and calibrated target-window probability at budgets 10, 25, 50, and 100. True labels are supplied only after ranking. Li, Zn, Ga, Se, and Te element-holdout studies are separately labeled custom extrapolation results.

## 4. Results

No complete result is available. `development-v1` was refused at data validation after pageouts increased by 2. An engineering repair eliminated redundant parsing without changing the frozen scientific protocol or resource gate. `development-v1-r2` then serialized and reload-checked two official fold checkpoints, but pageouts increased by 39 and the run was refused after fold 1. On 20 October 2026, `development-v1-r4` reload-validated its fold-0 checkpoint with peak RSS 163.9 MiB, but pageouts increased by 41 and the run was refused before fold 1. A fresh `development-v1-r3` attempt later that day passed instant preflight but was refused at data validation after pageouts increased by 39. Partial folds are intentionally not summarized.

## 5. Discussion

The execution failure is itself a bounded resource result: the frozen study was not admitted in the observed system state. It would be scientifically invalid to weaken the gate, reduce the model after observing failures, or report partial-fold estimates as final. The duplicate audit also shows why a standard benchmark split should not automatically be assumed composition-leakage-free.

## 6. Reproducibility and status

The data, manifests, protocol, source, lockfile, tests, failed run receipts, tracebacks, and partial checkpoints are retained. Numerical claims remain pending a new run ID in an external system state that satisfies the unchanged pageout gate.

## References

1. Dunn A, Wang Q, Ganose A, Dopp D, Jain A. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. *npj Computational Materials* 6, 138 (2020). https://doi.org/10.1038/s41524-020-00406-3
2. Zhuo Y, Mansouri Tehrani A, Brgoch J. Predicting the Band Gaps of Inorganic Solids by Machine Learning. *J. Phys. Chem. Lett.* 9, 1668–1673 (2018). https://doi.org/10.1021/acs.jpclett.8b00124

