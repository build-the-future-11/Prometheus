#!/usr/bin/env python3
"""Independently recompute v3 metrics, selection, and admission evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sparking.evidence import verify_manifest, write_json
from sparking.metrics import regression_metrics


METRICS = ("mae_ev", "rmse_ev", "median_ae_ev", "bias_ev")


def tree_name(exponent: float, aggregation: str) -> str:
    return f"extra_trees_alpha{float(exponent):.2f}_{aggregation}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    protocol_path = args.protocol or Path(__file__).resolve().parents[1] / "protocols" / "development-v3.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = verify_manifest(run_dir)
    results = json.loads((run_dir / "results.json").read_text())
    if results["status"] != "SUCCESS":
        raise ValueError("run did not finish successfully")
    if results["protected_confirmatory_outcomes_observed"] or results["official_fold_outcomes_observed"]:
        raise ValueError("a development run observed protected outcomes")

    comparisons = 0
    rows = results["rows"]
    for element in protocol["development_elements"]:
        with np.load(run_dir / f"{element}-predictions.npz") as arrays:
            truth = arrays["target_ev"]
            element_rows = [row for row in rows if row["element"] == element]
            if len(element_rows) != 10:
                raise ValueError(f"unexpected model row count for {element}: {len(element_rows)}")
            for row in element_rows:
                key = f"prediction_{row['model']}"
                if key not in arrays:
                    raise ValueError(f"missing raw prediction array {key}")
                observed = regression_metrics(truth, arrays[key])
                for metric in METRICS:
                    if not np.isclose(float(row[metric]), float(observed[metric]), rtol=0.0, atol=1e-10):
                        raise ValueError(f"metric drift for {element}/{row['model']}/{metric}")
                    comparisons += 1
            for exponent_value in protocol["frequency_balance"]["exponents"]:
                exponent = float(exponent_value)
                weights = arrays[f"training_weight_alpha{exponent:.2f}"]
                if weights.size != int(element_rows[0]["train_n"]):
                    raise ValueError(f"training weight length drift for {element}/alpha={exponent}")
                if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                    raise ValueError(f"invalid weights for {element}/alpha={exponent}")
                if not np.isclose(float(np.mean(weights)), 1.0, rtol=0.0, atol=1e-12):
                    raise ValueError(f"weight normalization drift for {element}/alpha={exponent}")
                if exponent == 0.0 and not np.array_equal(weights, np.ones(weights.size)):
                    raise ValueError(f"alpha-zero control is not exactly unweighted for {element}")
                if exponent > 0.0 and float(np.std(weights)) <= 1e-12:
                    raise ValueError(f"frequency weights are degenerate for {element}/alpha={exponent}")

    candidates = []
    for exponent_value in protocol["frequency_balance"]["candidate_exponents"]:
        exponent = float(exponent_value)
        for aggregation in protocol["aggregations"]:
            name = tree_name(exponent, str(aggregation))
            matching = [row for row in rows if row["model"] == name]
            candidates.append(
                {
                    "model": name,
                    "frequency_exponent": exponent,
                    "aggregation": aggregation,
                    "macro_mae_ev": float(np.mean([float(row["mae_ev"]) for row in matching])),
                }
            )
    minimum = min(float(candidate["macro_mae_ev"]) for candidate in candidates)
    eligible = [candidate for candidate in candidates if float(candidate["macro_mae_ev"]) <= minimum + 0.003]
    order = {"mean": 0, "median": 1}
    selected = sorted(
        eligible,
        key=lambda item: (float(item["frequency_exponent"]), order[str(item["aggregation"])]),
    )[0]
    recorded_selection = results["selection"]
    if recorded_selection["candidates"] != candidates or recorded_selection["selected"] != selected:
        raise ValueError("candidate selection drift")

    selected_rows = {row["element"]: row for row in rows if row["model"] == selected["model"]}
    control_name = tree_name(0.0, str(selected["aggregation"]))
    control_rows = {row["element"]: row for row in rows if row["model"] == control_name}
    catboost_rows = {row["element"]: row for row in rows if row["model"] == "catboost_ensemble"}
    elements = protocol["development_elements"]
    unweighted_improvements = {
        element: float(control_rows[element]["mae_ev"]) - float(selected_rows[element]["mae_ev"])
        for element in elements
    }
    catboost_improvements = {
        element: float(catboost_rows[element]["mae_ev"]) - float(selected_rows[element]["mae_ev"])
        for element in elements
    }
    macro_unweighted = float(np.mean(list(unweighted_improvements.values())))
    macro_catboost = float(np.mean(list(catboost_improvements.values())))
    improved_count = int(sum(value > 0.0 for value in unweighted_improvements.values()))
    admission = protocol["development_admission_rule"]
    expected_pass = (
        float(selected["frequency_exponent"]) > 0.0
        and macro_unweighted
        > float(admission["minimum_macro_mae_improvement_over_matched_unweighted_extra_trees_ev"])
        and macro_catboost > float(admission["minimum_macro_mae_improvement_over_catboost_ev"])
        and improved_count
        >= int(admission["minimum_improved_elements_over_matched_unweighted_extra_trees"])
    )
    recorded_admission = results["development_admission"]
    if not np.isclose(
        macro_unweighted,
        float(recorded_admission["macro_mae_improvement_over_matched_unweighted_extra_trees_ev"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("unweighted-control admission effect drift")
    if not np.isclose(
        macro_catboost,
        float(recorded_admission["macro_mae_improvement_over_catboost_ev"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("CatBoost admission effect drift")
    if improved_count != int(recorded_admission["improved_element_count_over_matched_unweighted_extra_trees"]):
        raise ValueError("improved-element count drift")
    if bool(recorded_admission["passed"]) != expected_pass:
        raise ValueError("admission decision drift")
    expected_disposition = "DEVELOPMENT_ADMITTED" if expected_pass else "DEVELOPMENT_NOT_SUPPORTED"
    if results["scientific_disposition"] != expected_disposition:
        raise ValueError("scientific disposition drift")

    receipt = json.loads((run_dir / "receipt.json").read_text())
    gpu = json.loads((run_dir / "gpu-admission.json").read_text())
    verification = {
        "status": "VERIFIED",
        "manifest": manifest,
        "metric_comparisons": comparisons,
        "selected_model": selected["model"],
        "scientific_disposition": expected_disposition,
        "protected_outcomes_observed": False,
        "run_receipt_status": receipt["status"],
        "gpu_admitted": gpu["exit_code"] == 0 and bool(gpu["stdout"]),
    }
    destination = args.output or run_dir.parent / f"{run_dir.name}-verification.json"
    write_json(destination, verification)
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
