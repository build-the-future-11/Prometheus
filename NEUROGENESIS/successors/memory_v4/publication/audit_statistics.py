"""Independent standard-library reconstruction of the registered v4 analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def read(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def percentile_bootstrap(values, seed, draws):
    rng = random.Random(seed)
    sampled = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [sampled[int(0.025 * draws)], sampled[int(0.975 * draws) - 1]]


def close(left, right, tolerance=1e-15):
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def main(run: Path, lock_path: Path, analysis_path: Path, output: Path) -> None:
    lock = read(lock_path)
    protocol = lock["protocol"]
    seeds = protocol["confirmation_seeds"]
    primary = protocol["primary_conditions"]
    draws = protocol["bootstrap_draws"]
    seed = protocol["bootstrap_seed"]
    analysis = read(analysis_path)
    rows = [read(path) for path in sorted((run / "attempts").glob("*/result.json"))]
    lookup = {(row["task_id"], row["method"]): row for row in rows}
    comparisons = {}
    errors = []
    for comparator in ("baseline", "all_update", "no_bank", "periodic", "replay", "knn", "frozen"):
        effects = []
        reductions = []
        for item_seed in seeds:
            candidate = statistics.mean(
                lookup[(f"{condition}-{item_seed}", "candidate")]["mse"] for condition in primary
            )
            control = statistics.mean(
                lookup[(f"{condition}-{item_seed}", comparator)]["mse"] for condition in primary
            )
            effects.append(candidate / max(control, 1e-30) - 1)
            candidate_updates = sum(
                lookup[(f"{condition}-{item_seed}", "candidate")]["usage"]["updates"]
                for condition in primary
            )
            control_updates = sum(
                lookup[(f"{condition}-{item_seed}", comparator)]["usage"]["updates"]
                for condition in primary
            )
            reductions.append(1 - candidate_updates / control_updates if control_updates else None)
        valid_reductions = [value for value in reductions if value is not None]
        reconstructed = {
            "relative_mse_change": statistics.mean(effects),
            "ci95": percentile_bootstrap(effects, seed, draws),
            "per_seed_effects": effects,
            "mean_update_reduction": statistics.mean(valid_reductions) if valid_reductions else None,
        }
        comparisons[comparator] = reconstructed
        reported = analysis["comparisons"][comparator]
        for key in ("relative_mse_change", "mean_update_reduction"):
            if reconstructed[key] is None and reported[key] is None:
                continue
            if reconstructed[key] is None or reported[key] is None or not close(reconstructed[key], reported[key]):
                errors.append(f"{comparator}.{key}")
        for index, (left, right) in enumerate(zip(reconstructed["ci95"], reported["ci95"])):
            if not close(left, right):
                errors.append(f"{comparator}.ci95[{index}]")
        for index, (left, right) in enumerate(zip(effects, reported["per_seed_effects"])):
            if not close(left, right):
                errors.append(f"{comparator}.per_seed_effects[{index}]")

    primary_result = comparisons["baseline"]
    reconstructed_disposition = (
        "SUPPORTED"
        if primary_result["ci95"][1] <= protocol["noninferiority_margin"]
        and primary_result["mean_update_reduction"] >= protocol["minimum_update_reduction"]
        else "NEGATIVE"
        if primary_result["ci95"][0] > protocol["noninferiority_margin"]
        or primary_result["mean_update_reduction"] < protocol["minimum_update_reduction"]
        else "INCONCLUSIVE"
    )
    if reconstructed_disposition != analysis["disposition"]:
        errors.append("disposition")
    report = {
        "status": "VALID" if not errors else "INVALID",
        "failed": errors,
        "attempts": len(rows),
        "independent_units": len(seeds),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "comparisons": comparisons,
        "disposition": reconstructed_disposition,
        "implementation_independence": "does not import experiment.py, engine.py, or audit.py",
        "independent_external_replication": False,
    }
    write(output, report)
    if errors:
        raise ValueError(f"statistical audit failed: {errors}")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.run, arguments.lock, arguments.analysis, arguments.output)
