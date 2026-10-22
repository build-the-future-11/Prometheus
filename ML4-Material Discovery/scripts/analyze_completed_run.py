"""Read-only, post-hoc statistical audit for a completed development run.

This analysis does not alter the frozen run or promote it to confirmatory
evidence. Bootstrap resampling uses canonical compositions as independent
units so the three duplicate groups are not treated as extra evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scripts.reproduce_submission import verify_release_candidate
except ModuleNotFoundError:  # Direct `python scripts/analyze_completed_run.py` execution.
    from reproduce_submission import verify_release_candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _bootstrap_means(
    values: np.ndarray, *, repetitions: int, seed: int, chunk_size: int = 256
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        result[start:stop] = values[indices].mean(axis=1)
    return result


def _hypergeometric_tail(
    *, population: int, successes: int, draws: int, observed: int
) -> float:
    upper = min(draws, successes)
    terms = []
    denominator = math.lgamma(population + 1) - math.lgamma(draws + 1) - math.lgamma(
        population - draws + 1
    )
    for hits in range(observed, upper + 1):
        if draws - hits > population - successes:
            continue
        log_probability = (
            math.lgamma(successes + 1)
            - math.lgamma(hits + 1)
            - math.lgamma(successes - hits + 1)
            + math.lgamma(population - successes + 1)
            - math.lgamma(draws - hits + 1)
            - math.lgamma(population - successes - (draws - hits) + 1)
            - denominator
        )
        terms.append(math.exp(log_probability))
    return float(sum(terms))


def _group_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["canonical_composition"], []).append(row)
    return grouped


def analyze(root: Path, run_id: str, repetitions: int, seed: int) -> dict:
    errors = verify_release_candidate(root, run_id)
    if errors:
        raise RuntimeError("release gate refused run: " + "; ".join(errors))
    run_dir = root / "runs" / run_id
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    with (run_dir / "predictions/oof_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        grouped = _group_rows(csv.DictReader(handle))

    proposed_errors = []
    ridge_errors = []
    coverage_90 = []
    for group in grouped.values():
        proposed_errors.append(
            np.mean(
                [
                    abs(float(row["target_gap_ev"]) - float(row["proposed_prediction_ev"]))
                    for row in group
                ]
            )
        )
        ridge_errors.append(
            np.mean(
                [
                    abs(float(row["target_gap_ev"]) - float(row["ridge_prediction_ev"]))
                    for row in group
                ]
            )
        )
        coverage_90.append(
            np.mean(
                [
                    float(row["interval_90_lower_ev"])
                    <= float(row["target_gap_ev"])
                    <= float(row["interval_90_upper_ev"])
                    for row in group
                ]
            )
        )

    proposed = np.asarray(proposed_errors, dtype=float)
    ridge = np.asarray(ridge_errors, dtype=float)
    differences = proposed - ridge
    coverage = np.asarray(coverage_90, dtype=float)
    proposed_bootstrap = _bootstrap_means(
        proposed, repetitions=repetitions, seed=seed
    )
    difference_bootstrap = _bootstrap_means(
        differences, repetitions=repetitions, seed=seed + 1
    )
    coverage_bootstrap = _bootstrap_means(
        coverage, repetitions=repetitions, seed=seed + 2
    )

    ranking = results["candidate_ranking"]
    budget_50 = {
        row["policy"]: row
        for row in ranking["metrics"]
        if int(row["budget"]) == 50
    }
    population = int(ranking["pool_size"])
    successes = int(round(float(ranking["pool_prevalence"]) * population))
    proposed_hits = int(budget_50["proposed_calibrated_probability"]["hits"])

    return {
        "schema_version": 1,
        "analysis_status": "POST_HOC_EXPLORATORY_NOT_CONFIRMATORY",
        "run_id": run_id,
        "independent_unit": "canonical_composition",
        "independent_unit_count": len(grouped),
        "bootstrap": {"repetitions": repetitions, "seed": seed, "interval": "percentile_95"},
        "source_hashes": {
            "results_json_sha256": _sha256(run_dir / "results.json"),
            "oof_predictions_csv_sha256": _sha256(
                run_dir / "predictions/oof_predictions.csv"
            ),
            "artifact_manifest_sha256": _sha256(run_dir / "artifact_manifest.json"),
        },
        "official_split_prediction": {
            "group_weighted_proposed_mae_ev": float(proposed.mean()),
            "group_weighted_proposed_mae_95_interval_ev": _percentile_interval(
                proposed_bootstrap
            ),
            "group_weighted_mae_difference_proposed_minus_ridge_ev": float(
                differences.mean()
            ),
            "group_weighted_mae_difference_95_interval_ev": _percentile_interval(
                difference_bootstrap
            ),
            "bootstrap_probability_proposed_mae_below_ridge": float(
                np.mean(difference_bootstrap < 0)
            ),
            "relative_mae_reduction_vs_ridge": float(
                1.0 - proposed.mean() / ridge.mean()
            ),
        },
        "uncertainty": {
            "group_weighted_90_coverage": float(coverage.mean()),
            "group_weighted_90_coverage_95_interval": _percentile_interval(
                coverage_bootstrap
            ),
            "nominal_coverage": 0.9,
        },
        "ranking_budget_50": {
            "population": population,
            "target_window_hits": successes,
            "proposed_hits": proposed_hits,
            "proposed_precision": float(
                budget_50["proposed_calibrated_probability"]["precision"]
            ),
            "greedy_hits": int(budget_50["greedy_predicted_property"]["hits"]),
            "greedy_precision": float(
                budget_50["greedy_predicted_property"]["precision"]
            ),
            "random_mean_hits_64_replicates": float(budget_50["random"]["hits"]),
            "random_selection_probability_of_at_least_proposed_hits": _hypergeometric_tail(
                population=population,
                successes=successes,
                draws=50,
                observed=proposed_hits,
            ),
        },
        "interpretation_boundary": (
            "Intervals and tail probabilities were computed after observing the development run. "
            "They quantify this retained benchmark artifact and do not convert it into a frozen "
            "confirmatory, prospective, or independently replicated study."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    payload = analyze(args.root.resolve(), args.run_id, args.repetitions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
