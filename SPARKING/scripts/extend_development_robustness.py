#!/usr/bin/env python3
"""Retrospective domain sensitivity of the frozen v3 material predictions.

This script never fits a model or opens protected elements. It keeps the original
admission decision and computes exact finite-panel reweighting sensitivities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sparking.evidence import verify_manifest
from sparking.metrics import regression_metrics


def worst_reweighting(values: np.ndarray, tv_budget: float) -> tuple[float, np.ndarray]:
    """Minimize q @ values subject to TV(q, uniform) <= tv_budget.

    Move mass from the largest value to the smallest, using each donor once.
    The sorted transfer rule solves this finite linear program exactly.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("reweighting needs at least two finite domain effects")
    if not np.isfinite(tv_budget) or not 0 <= tv_budget <= 1:
        raise ValueError("TV budget must be in [0,1]")
    weights = np.full(values.size, 1.0 / values.size)
    receiver = int(np.argmin(values))
    remaining = min(float(tv_budget), 1.0 - weights[receiver])
    for donor in np.argsort(values)[::-1]:
        if donor == receiver or remaining <= 0.0:
            continue
        transfer = min(float(weights[donor]), remaining)
        weights[donor] -= transfer
        weights[receiver] += transfer
        remaining -= transfer
    return float(weights @ values), weights


def select_candidate(scores: dict[str, float], candidates: list[dict], tie: float) -> str:
    minimum = min(scores[item["model"]] for item in candidates)
    eligible = [item for item in candidates if scores[item["model"]] <= minimum + tie]
    return min(eligible, key=lambda item: (item["frequency_exponent"], item["aggregation"] != "mean"))["model"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run: Path, protocol_path: Path) -> dict:
    manifest = verify_manifest(run)
    results = json.loads((run / "results.json").read_text())
    protocol = json.loads(protocol_path.read_text())
    elements = protocol["development_elements"]
    if results["protected_confirmatory_outcomes_observed"] or results["official_fold_outcomes_observed"]:
        raise ValueError("saved run is not an unprotected development analysis")
    selected = results["selection"]["selected"]["model"]
    control = results["development_admission"]["matched_unweighted_control"]
    candidates = results["selection"]["candidates"]
    row_lookup = {(row["element"], row["model"]): row for row in results["rows"]}
    if len(row_lookup) != len(results["rows"]):
        raise ValueError("duplicate element/model metrics")
    arrays_by_element, scores, deltas, row_sets, target_by_id = {}, {}, {}, {}, {}
    comparisons = 0
    for element in elements:
        with np.load(run / f"{element}-predictions.npz", allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
        arrays_by_element[element] = arrays
        target = arrays["target_ev"]
        row_ids = arrays["row_index"]
        if row_ids.dtype.kind not in "iu" or row_ids.shape != target.shape or len(np.unique(row_ids)) != len(row_ids):
            raise ValueError(f"invalid row identities for {element}")
        row_sets[element] = set(map(int, row_ids))
        for row_id, value in zip(row_ids, target, strict=True):
            previous = target_by_id.setdefault(int(row_id), float(value))
            if previous != float(value):
                raise ValueError("overlapping row has inconsistent ground truth")
        scores[element] = {}
        for (recorded_element, model), row in row_lookup.items():
            if recorded_element != element:
                continue
            metrics = regression_metrics(target, arrays[f"prediction_{model}"])
            for name in ("mae_ev", "rmse_ev", "median_ae_ev", "bias_ev"):
                if not np.isclose(metrics[name], row[name], rtol=0.0, atol=1e-10):
                    raise ValueError(f"metric drift: {element}/{model}/{name}")
                comparisons += 1
            scores[element][model] = metrics["mae_ev"]
        deltas[element] = scores[element][control] - scores[element][selected]
    effects = np.asarray([deltas[element] for element in elements])
    macro = float(effects.mean())
    recorded = results["development_admission"]["macro_mae_improvement_over_matched_unweighted_extra_trees_ev"]
    if not np.isclose(macro, recorded, atol=1e-12, rtol=0.0):
        raise ValueError("macro effect drift")
    by_model = {model: float(np.mean([scores[e][model] for e in elements])) for model in scores[elements[0]]}
    if select_candidate(by_model, candidates, results["selection"]["tie_band_ev"]) != selected:
        raise ValueError("selection rule drift")

    removal = []
    for element in elements:
        remaining = [other for other in elements if other != element]
        candidates_on_remainder = {item["model"]: float(np.mean([scores[e][item["model"]] for e in remaining])) for item in candidates}
        replacement = select_candidate(candidates_on_remainder, candidates, results["selection"]["tie_band_ev"])
        aggregation = "mean" if replacement.endswith("_mean") else "median"
        heldout_control = f"extra_trees_alpha0.00_{aggregation}"
        removal.append({
            "omitted_element": element,
            "selected_model_remainder_improvement_ev": float(np.mean([deltas[e] for e in remaining])),
            "candidate_selected_using_remainder": replacement,
            "omitted_element_improvement_for_reselected_candidate_ev": scores[element][heldout_control] - scores[element][replacement],
        })

    weighted = []
    for budget in (0.0, 0.05, 0.10, 0.125, 0.20, 0.25):
        value, weights = worst_reweighting(effects, budget)
        weighted.append({"total_variation_budget": budget, "minimum_weighted_improvement_ev": value, "domain_weights": dict(zip(elements, map(float, weights), strict=True))})
    low, high = 0.0, 1.0
    if macro > 0 and effects.min() < 0:
        for _ in range(60):
            middle = (low + high) / 2
            if worst_reweighting(effects, middle)[0] > 0:
                low = middle
            else:
                high = middle
        zero_tv = high
    else:
        zero_tv = 0.0 if macro <= 0 else None

    exponent = results["selection"]["selected"]["frequency_exponent"]
    seed_results = []
    for seed_index, seed in enumerate(protocol["extra_trees"]["seeds"]):
        effect_by_element = {}
        for element in elements:
            arrays = arrays_by_element[element]
            y = arrays["target_ev"]
            a = arrays[f"extra_trees_members_alpha{exponent:.2f}_ev"][seed_index]
            b = arrays["extra_trees_members_alpha0.00_ev"][seed_index]
            effect_by_element[element] = float(np.mean(np.abs(y-b) - np.abs(y-a)))
        seed_results.append({"seed": seed, "paired_single_seed_macro_improvement_ev": float(np.mean(list(effect_by_element.values()))), "per_element": effect_by_element})

    largest = elements[int(np.argmax(effects))]
    return {
        "schema_version": 1,
        "analysis_status": "RETROSPECTIVE_FIXED_PANEL_SENSITIVITY",
        "new_model_fits": 0,
        "primary_disposition": results["scientific_disposition"],
        "original_admission_passed": results["development_admission"]["passed"],
        "protected_outcomes_observed": False,
        "interpretation": "Exact diagnostics for these eight already-observed domains. Domain removal and seed reuse are descriptive; overlapping training/evaluation samples prevent interpreting these as independent confirmatory tests.",
        "selected_model": selected,
        "matched_control": control,
        "selected_macro_improvement_ev": macro,
        "per_element_improvement_ev": deltas,
        "positive_domains": int(np.count_nonzero(effects > 0)),
        "total_domains": len(elements),
        "largest_contribution": {"element": largest, "effect_ev": deltas[largest], "fraction_of_net_summed_improvement": float(effects.max() / effects.sum())},
        "removing_one_domain": removal,
        "reweighting_sensitivity": weighted,
        "minimum_total_variation_to_nonpositive_effect": zero_tv,
        "paired_training_seed_sensitivity": seed_results,
        "overlapping_row_counts": {a: {b: len(row_sets[a] & row_sets[b]) for b in elements} for a in elements},
        "unique_row_count": len(target_by_id),
        "total_evaluation_rows_with_reuse": sum(map(len, row_sets.values())),
        "verification": {"manifest": manifest, "metric_comparisons": comparisons, "row_identity_target_consistency": True},
        "source_sha256": {"results.json": digest(run/"results.json"), "protocol": digest(protocol_path), **{f"{e}-predictions.npz": digest(run/f"{e}-predictions.npz") for e in elements}},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("analysis_status", "selected_macro_improvement_ev", "largest_contribution", "minimum_total_variation_to_nonpositive_effect", "verification")}, indent=2))


if __name__ == "__main__":
    main()
