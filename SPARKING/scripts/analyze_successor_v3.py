"""Deterministic post-hoc uncertainty analysis for verified SPARKING v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ELEMENTS = ("O", "S", "In", "Ge", "Sb", "Ba", "Co", "F")
BOOTSTRAP_SEED = 20261022
BOOTSTRAP_REPLICATES = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: np.ndarray, threshold: float) -> dict[str, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "lower_95_ev": float(lower),
        "upper_95_ev": float(upper),
        "probability_improvement_gt_0": float(np.mean(values > 0.0)),
        "probability_improvement_gt_threshold": float(np.mean(values > threshold)),
    }


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    batch = 500
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        output[start:stop] = values[indices].mean(axis=1)
    return output


def cluster_bootstrap(
    row_ids: dict[str, np.ndarray], effects: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    unique = np.unique(np.concatenate([row_ids[element] for element in ELEMENTS]))
    lookup = {int(row): index for index, row in enumerate(unique)}
    sums = np.zeros((len(unique), len(ELEMENTS)), dtype=float)
    present = np.zeros_like(sums)
    for column, element in enumerate(ELEMENTS):
        for row, effect in zip(row_ids[element], effects[element], strict=True):
            index = lookup[int(row)]
            sums[index, column] += float(effect)
            present[index, column] += 1.0
    output = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    probabilities = np.full(len(unique), 1.0 / len(unique))
    batch = 200
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        counts = rng.multinomial(len(unique), probabilities, size=stop - start)
        numerators = counts @ sums
        denominators = counts @ present
        if np.any(denominators == 0):
            raise RuntimeError("cluster bootstrap produced an empty element scenario")
        output[start:stop] = (numerators / denominators).mean(axis=1)
    return output, len(unique)


def comparison(
    run: Path,
    selected: str,
    baseline: str,
    declared: dict[str, float],
    threshold: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    row_ids: dict[str, np.ndarray] = {}
    effects: dict[str, np.ndarray] = {}
    per_element: dict[str, object] = {}
    for element in ELEMENTS:
        with np.load(run / f"{element}-predictions.npz") as arrays:
            truth = arrays["target_ev"]
            improvement = np.abs(truth - arrays[f"prediction_{baseline}"]) - np.abs(
                truth - arrays[f"prediction_{selected}"]
            )
            point = float(np.mean(improvement))
            if not np.isclose(point, float(declared[element]), atol=1e-12, rtol=0.0):
                raise RuntimeError(f"raw effect drift for {element}/{baseline}")
            row_ids[element] = arrays["row_index"].astype(np.int64)
            effects[element] = improvement
            per_element[element] = {
                "n": int(len(truth)),
                "paired_mae_improvement_ev": point,
                "paired_row_bootstrap": percentile(bootstrap_mean(improvement, rng), threshold),
            }
    macro = float(np.mean([float(declared[element]) for element in ELEMENTS]))
    cluster, unique_count = cluster_bootstrap(row_ids, effects, rng)
    return {
        "baseline_model": baseline,
        "preregistered_threshold_ev": threshold,
        "macro_mae_improvement_ev": macro,
        "margin_above_threshold_ev": macro - threshold,
        "unique_cluster_count": unique_count,
        "cluster_bootstrap": percentile(cluster, threshold),
        "per_element": per_element,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    results_path = run / "results.json"
    results = json.loads(results_path.read_text())
    if results["scientific_disposition"] != "DEVELOPMENT_NOT_SUPPORTED":
        raise RuntimeError("unexpected primary disposition")
    selected = str(results["selection"]["selected"]["model"])
    admission = results["development_admission"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows_by_model: dict[str, dict[str, float]] = {}
    for row in results["rows"]:
        rows_by_model.setdefault(str(row["model"]), {})[str(row["element"])] = float(row["mae_ev"])
    model_macro = [
        {
            "model": model,
            "per_element_mae_ev": values,
            "macro_mae_ev": float(np.mean([values[element] for element in ELEMENTS])),
        }
        for model, values in rows_by_model.items()
    ]
    model_macro.sort(key=lambda item: float(item["macro_mae_ev"]))

    output = {
        "schema_version": 1,
        "run_id": results["run_id"],
        "analysis_status": "VERIFIED_POST_HOC_DESCRIPTIVE",
        "primary_disposition": results["scientific_disposition"],
        "protected_outcomes_observed": False,
        "selected_model": selected,
        "bootstrap": {
            "status": "POST_HOC_NOT_PREREGISTERED",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "cluster_unit": "unique Matbench row_index across overlapping element holdouts",
        },
        "comparisons": {
            "matched_unweighted_extra_trees": comparison(
                run,
                selected,
                str(admission["matched_unweighted_control"]),
                admission["per_element_improvement_over_matched_unweighted_extra_trees_ev"],
                float(
                    admission["admission_rule"]
                    ["minimum_macro_mae_improvement_over_matched_unweighted_extra_trees_ev"]
                ),
                rng,
            ),
            "catboost_ensemble": comparison(
                run,
                selected,
                "catboost_ensemble",
                admission["per_element_improvement_over_catboost_ev"],
                float(admission["admission_rule"]["minimum_macro_mae_improvement_over_catboost_ev"]),
                rng,
            ),
        },
        "model_macro_mae": model_macro,
        "source_files": {
            "results_json_sha256": sha256(results_path),
            **{
                f"{element}_predictions_npz_sha256": sha256(run / f"{element}-predictions.npz")
                for element in ELEMENTS
            },
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
