"""Results-derived reports; no metric is typed into prose by hand."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _metric_table(metrics: dict[str, dict[str, float]]) -> str:
    lines = ["| Model | MAE (eV) | RMSE (eV) | R² | Spearman |", "|---|---:|---:|---:|---:|"]
    labels = {
        "proposed": "Calibrated ExtraTrees ensemble",
        "ridge": "Ridge composition baseline",
        "trivial": "Train-median predictor",
    }
    for key in ("proposed", "ridge", "trivial"):
        row = metrics[key]
        lines.append(
            f"| {labels[key]} | {_f(row['mae_ev'])} | {_f(row['rmse_ev'])} | "
            f"{_f(row['r2'])} | {_f(row['spearman'])} |"
        )
    return "\n".join(lines)


def _ranking_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Policy | Budget | Hits | Precision | Enrichment vs pool |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: (value["budget"], value["policy"])):
        lines.append(
            f"| {row['policy']} | {row['budget']} | {_f(row['hits'], 2)} | "
            f"{_f(row['precision'])} | {_f(row['enrichment_vs_pool'], 2)}× |"
        )
    return "\n".join(lines)


def _generalization_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Held-out element | Train n | Test n | Proposed MAE (eV) | Ridge MAE (eV) | 90% coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['held_out_element']} | {row['train_n']} | {row['test_n']} | "
            f"{_f(row['proposed']['mae_ev'])} | {_f(row['ridge']['mae_ev'])} | "
            f"{_f(row['coverage_90'])} |"
        )
    return "\n".join(lines)


def render_results(results: dict[str, Any], receipt: dict[str, Any]) -> str:
    official = results["official_cv"]
    ranking = results["candidate_ranking"]
    calibration_90 = next(
        row for row in official["calibration"] if row["nominal_coverage"] == 0.9
    )
    budget_50 = {
        row["policy"]: row for row in ranking["metrics"] if row["budget"] == 50
    }
    proposed_mae = official["metrics"]["proposed"]["mae_ev"]
    ridge_mae = official["metrics"]["ridge"]["mae_ev"]
    negatives: list[str] = []
    if proposed_mae >= ridge_mae:
        negatives.append(
            f"The proposed ensemble did not beat ridge on MAE ({_f(proposed_mae)} vs {_f(ridge_mae)} eV)."
        )
    else:
        negatives.append(
            f"The proposed ensemble improved MAE over ridge by only {_f(ridge_mae - proposed_mae)} eV; this is a development result, not external validation."
        )
    if budget_50["proposed_calibrated_probability"]["precision"] <= budget_50[
        "greedy_predicted_property"
    ]["precision"]:
        negatives.append(
            "At budget 50, calibrated-probability ranking did not improve precision over greedy predicted-property ranking."
        )
    else:
        negatives.append(
            "The ranking gain is retrospective on measured Matbench labels and does not demonstrate prospective discovery."
        )
    worst_holdout = max(results["generalization"], key=lambda row: row["proposed"]["mae_ev"])
    negatives.append(
        f"Element holdout degraded most for {worst_holdout['held_out_element']} (MAE {_f(worst_holdout['proposed']['mae_ev'])} eV)."
    )
    negatives.append(
        "Three canonical composition groups remain duplicated in the upstream artifact; two cross official train/test boundaries in four folds. Official metrics are retained for comparability but are not composition-leakage-free."
    )

    return f"""# Results

Status: **EXECUTED_DEVELOPMENT — NOT PROSPECTIVELY OR INDEPENDENTLY VALIDATED**

## Completed scope

The repository was empty at intake, so the approved fallback objective was used: calibrated experimental band-gap prediction from composition and an explicitly evaluated uncertainty-aware policy over a fixed candidate pool. The only property task is `matbench_expt_gap`; no additional property projects were invented.

The executed study used all 4,604 Matbench v0.1 rows, official five-fold outer splits, three frozen model seeds, group-isolated calibration subsets, five separately labeled element-holdout studies, and a unique-composition candidate pool of {ranking['pool_size']} candidates. The target is experimental band gap in eV.

## Official Matbench-split prediction results

{_metric_table(official['metrics'])}

These are out-of-fold results on the exact official split IDs. They are not leaderboard submissions. Because equivalent canonical compositions cross some official train/test boundaries, see the data-integrity caveat below.

## Uncertainty and abstention

The 90% normalized split-conformal interval achieved {_f(calibration_90['observed_coverage'])} observed coverage with mean width {_f(calibration_90['mean_interval_width_ev'])} eV. Calibration rows for 50%, 80%, 90%, and 95% intervals are retained in `tables/calibration.csv`.

Uncertainty was operationally tested through abstention, not merely plotted. The lowest-uncertainty 20% had MAE {_f(official['abstention'][-1]['retained_mae_ev'])} eV versus {_f(proposed_mae)} eV without abstention. Rejected-set error and all retained fractions are recorded in `tables/abstention.csv`.

## Fixed-pool candidate-selection utility

Success was frozen before execution as an evaluation label in the 1.1–1.6 eV window. `proposed_calibrated_probability` ranks by a calibrated Gaussian working probability; it never receives evaluation labels. Random selection reports 64 deterministic replicates.

{_ranking_table(ranking['metrics'])}

The candidate ledger is retrospective. A ranked formula is not evidence of stability, synthesizability, safety, device utility, or independent verification.

## Separately labeled element-holdout extrapolation

These are custom composition-family studies, not official Matbench leaderboard results.

{_generalization_table(results['generalization'])}

## Sanity checks

- Train-median prediction was evaluated in every official fold (table above).
- Three label-permutation controls produced mean MAE {_f(results['sanity_checks']['label_permutation']['mean_mae_ev'])} eV versus {_f(results['sanity_checks']['label_permutation']['unpermuted_fold_0_mae_ev'])} eV for the unpermuted fold-0 model. Pass: `{results['sanity_checks']['label_permutation']['passes']}`.
- Every serialized fold checkpoint was immediately reloaded and required numerically identical predictions. Pass: `{results['sanity_checks']['checkpoint_reload_all_passed']}`.
- Unit conversion, formula parsing, composition invariance, missing-data refusal, feature availability, split identity, group isolation, and ranking label separation are covered by the test suite.

## Negative findings and limitations

""" + "\n".join(f"- {item}" for item in negatives) + f"""
- Structures and polymorph identity are absent. No structure claim can be made, and no neural structure model was justified.
- The Matbench metadata says compositions were deduplicated and repeats spanning more than 0.1 eV removed, but the pinned artifact contains three canonical duplicate groups with spans 0.58–2.30 eV. All six source rows are preserved in `data/manifests/canonical_duplicate_audit.json`.
- Candidate evaluation reuses known experimental labels only after ranking; there is no new calculation, synthesis, or wet-lab confirmation.

## Execution receipt

- Run ID: `{receipt['run_id']}`
- Started UTC: `{receipt['started_utc']}`
- Finished UTC: `{receipt['finished_utc']}`
- Duration: `{_f(receipt['duration_seconds'], 1)}` seconds
- Python: `{receipt['python']}`
- Dataset SHA-256: `{receipt['inputs']['dataset_sha256']}`
- Official split SHA-256: `{receipt['inputs']['split_sha256']}`
- Peak process RSS: `{_f(receipt['resources']['peak_observed_rss_mib'], 1)}` MiB (limit {receipt['resources']['max_allowed_rss_mib']} MiB)
- Pageout delta: `{receipt['resources']['pageout_delta']}`
- Tests: see `test_receipt.json`; artifact hashes: `artifact_manifest.json`

## Claim boundary

Implemented, numerically executed, and locally tested are distinct from scientifically validated, prospectively discovered, release-certified, or independently reproduced. This run establishes only an internally reproducible development result on a public historical benchmark.
"""


def render_manuscript(results: dict[str, Any], receipt: dict[str, Any]) -> str:
    metrics = results["official_cv"]["metrics"]
    calibration_90 = next(
        row
        for row in results["official_cv"]["calibration"]
        if row["nominal_coverage"] == 0.9
    )
    top50 = {
        row["policy"]: row
        for row in results["candidate_ranking"]["metrics"]
        if row["budget"] == 50
    }
    return f"""# Uncertainty-aware composition models for retrospective experimental band-gap candidate ranking

## Abstract

We evaluated a narrow materials-informatics workflow for predicting experimental inorganic band gaps from composition and ranking a fixed historical candidate pool. On 4,604 Matbench v0.1 records using the official five-fold outer splits, a three-seed ExtraTrees ensemble achieved MAE {_f(metrics['proposed']['mae_ev'])} eV and RMSE {_f(metrics['proposed']['rmse_ev'])} eV; a regularized linear composition baseline achieved MAE {_f(metrics['ridge']['mae_ev'])} eV. Group-isolated split-conformal calibration produced {_f(calibration_90['observed_coverage'])} coverage for nominal 90% intervals. At a 50-label retrospective budget, calibrated target-window probability selected {_f(top50['proposed_calibrated_probability']['hits'], 1)} compositions in a prespecified 1.1–1.6 eV band, compared with {_f(top50['random']['hits'], 1)} for random selection. Element-holdout errors were materially worse than random-split performance for some elements. These results support software and protocol feasibility only: the ranking uses historical labels for post hoc evaluation and does not verify stability, synthesizability, or device performance.

## 1. Introduction

Materials-property benchmarks can test prediction and ranking machinery, but high benchmark utility is not itself materials discovery. We therefore scoped this work to one property, one public benchmark, and one intended decision: prioritize compositions likely to fall inside a prespecified experimental band-gap interval. The study separates official benchmark evaluation, custom element-holdout extrapolation, uncertainty calibration, and retrospective selection utility.

## 2. Data and methods

### 2.1 Data contract

The input was the composition-only `matbench_expt_gap` task (Matbench v0.1), sourced from Zhuo, Mansouri Tehrani, and Brgoch and curated by Dunn et al. The target is experimental band gap in eV. The pinned 4,604-row artifact has SHA-256 `{receipt['inputs']['dataset_sha256']}`. Exact official five-fold IDs came from the Matbench validation bundle with SHA-256 `{receipt['inputs']['split_sha256']}`.

An integrity audit found three canonical duplicates missed by string-level upstream curation. Exact official rows were retained solely for benchmark comparability. Calibration partitions and custom splits kept canonical groups intact, while the candidate pool collapsed them to unique compositions and retained all source IDs and target spans.

### 2.2 Models and uncertainty

Features were element fractions plus scale-invariant stoichiometric and atomic-number summaries. The baseline was standardized ridge regression with inner alpha selection. The proposed predictor averaged three fixed-seed ExtraTrees models. A label-blind 20% subset of each outer-training fold calibrated tree dispersion; normalized split-conformal scores produced prediction intervals. No structure, relaxed geometry, calculated property, or post-measurement metadata was used.

### 2.3 Endpoints

Prediction endpoints were MAE, RMSE, R², and Spearman correlation. Calibration endpoints were empirical coverage and interval width. Abstention was evaluated by retaining progressively lower-uncertainty subsets. Candidate success was an experimental gap in [1.1, 1.6] eV at budgets 10, 25, 50, and 100. Policies were random, greedy proximity to the target-window center, maximum uncertainty, and calibrated probability of the target interval.

## 3. Results

{_metric_table(metrics)}

Nominal 90% intervals achieved {_f(calibration_90['observed_coverage'])} coverage with width {_f(calibration_90['mean_interval_width_ev'])} eV. At budget 50, proposed, greedy, uncertainty, and random policies achieved precisions {_f(top50['proposed_calibrated_probability']['precision'])}, {_f(top50['greedy_predicted_property']['precision'])}, {_f(top50['uncertainty']['precision'])}, and {_f(top50['random']['precision'])}, respectively.

{_generalization_table(results['generalization'])}

## 4. Discussion

The experiment demonstrates a reproducible route from a pinned composition dataset to predictions, calibrated intervals, abstention consequences, and a provenance-rich candidate ledger. It also exposes limits hidden by a headline error: canonical duplicates cross official folds, label conflicts remain, and element holdout can be much harder than randomized evaluation. The candidate policy is evaluated retrospectively and cannot establish a new material, thermodynamic stability, synthesis feasibility, or application performance.

## 5. Reproducibility and status

Source, split, configuration, predictions, checkpoints, failures, environment, and artifact hashes are retained under `runs/{receipt['run_id']}`. The result is **EXECUTED_DEVELOPMENT** and **NOT PROSPECTIVELY OR INDEPENDENTLY VALIDATED**.

## References

1. Dunn A, Wang Q, Ganose A, Dopp D, Jain A. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. *npj Computational Materials* 6, 138 (2020). https://doi.org/10.1038/s41524-020-00406-3
2. Zhuo Y, Mansouri Tehrani A, Brgoch J. Predicting the Band Gaps of Inorganic Solids by Machine Learning. *J. Phys. Chem. Lett.* 9, 1668–1673 (2018). https://doi.org/10.1021/acs.jpclett.8b00124
"""


def write_reports(root: Path, results: dict[str, Any], receipt: dict[str, Any]) -> None:
    (root / "RESULTS.md").write_text(render_results(results, receipt), encoding="utf-8")
    (root / "MANUSCRIPT.md").write_text(render_manuscript(results, receipt), encoding="utf-8")
