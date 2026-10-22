#!/usr/bin/env python3
"""Execute the frozen SPARKING v3 development matrix and retain raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import catboost
import matminer
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from sparking.data import DATA_SHA256, load_data
from sparking.evidence import ResourceGuard, create_manifest, environment_record, verify_manifest, write_json
from sparking.metrics import regression_metrics
from sparking.successor import (
    catboost_ensemble_predictions,
    element_frequency_sample_weights,
    extra_trees_ensemble_predictions,
    formula_element_mask,
    make_publication_ridge,
    publication_features,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_record(root: Path, protocol_path: Path) -> dict[str, Any]:
    git_root = root.parent
    record: dict[str, Any] = {
        "protocol_path": protocol_path.relative_to(root).as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "dataset_sha256": DATA_SHA256,
    }
    if (git_root / ".git").exists():
        record["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=git_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        record["git_project_status"] = subprocess.run(
            ["git", "status", "--short", "--", root.name],
            cwd=git_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    else:
        record["git_commit"] = os.environ.get("SPARKING_SOURCE_COMMIT", "HOSTED_ARCHIVE")
        record["git_project_status"] = []
    return record


def fit_ridge_predict(
    features: np.ndarray, targets: np.ndarray, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    model = make_publication_ridge()
    model.fit(features[train], targets[train])
    return np.clip(np.asarray(model.predict(features[test]), dtype=float), 0.0, None)


def fit_hgb_squared_predict(
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(features[train])
    x_test = imputer.transform(features[test])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=int(config["max_iter"]),
        learning_rate=float(config["learning_rate"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=float(config["l2_regularization"]),
        random_state=int(config["random_state"]),
        early_stopping=False,
    )
    model.fit(x_train, targets[train])
    return np.clip(np.asarray(model.predict(x_test), dtype=float), 0.0, None)


def tree_name(exponent: float, aggregation: str) -> str:
    if aggregation not in {"mean", "median"}:
        raise ValueError(f"unknown aggregation: {aggregation}")
    return f"extra_trees_alpha{float(exponent):.2f}_{aggregation}"


def effective_sample_size(weights: np.ndarray) -> float:
    return float(np.sum(weights) ** 2 / np.sum(np.square(weights)))


def evaluate_outer_holdout(
    run_dir: Path,
    formulas: list[str],
    targets: np.ndarray,
    features: np.ndarray,
    element: str,
    protocol: dict[str, Any],
    task_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mask = formula_element_mask(formulas, element)
    train = np.flatnonzero(~mask)
    test = np.flatnonzero(mask)
    train_formulas = [formulas[int(index)] for index in train]

    ridge = fit_ridge_predict(features, targets, train, test)
    hgb = fit_hgb_squared_predict(features, targets, train, test, protocol["corrected_hgb"])
    catboost_prediction = catboost_ensemble_predictions(
        features[train],
        targets[train],
        features[test],
        protocol["catboost"],
        protocol["catboost"]["seeds"],
        task_type,
    )

    predictions: dict[str, np.ndarray] = {
        "ridge": ridge,
        "direct_hgb_squared": hgb,
        "catboost_single_seed": catboost_prediction.members[0],
        "catboost_ensemble": catboost_prediction.mean,
    }
    retained: dict[str, np.ndarray] = {
        "row_index": test,
        "target_ev": targets[test],
        "prediction_ridge": ridge,
        "prediction_direct_hgb_squared": hgb,
        "catboost_members_ev": catboost_prediction.members,
        "prediction_catboost_single_seed": catboost_prediction.members[0],
        "prediction_catboost_ensemble": catboost_prediction.mean,
    }
    weight_diagnostics: dict[str, dict[str, float]] = {}
    balance = protocol["frequency_balance"]
    for exponent_value in balance["exponents"]:
        exponent = float(exponent_value)
        weights = element_frequency_sample_weights(
            train_formulas,
            exponent,
            minimum=float(balance["minimum_weight"]),
            maximum=float(balance["maximum_weight"]),
        )
        forest = extra_trees_ensemble_predictions(
            features[train],
            targets[train],
            features[test],
            protocol["extra_trees"],
            protocol["extra_trees"]["seeds"],
            sample_weight=weights,
        )
        token = f"{exponent:.2f}"
        retained[f"training_weight_alpha{token}"] = weights
        retained[f"extra_trees_members_alpha{token}_ev"] = forest.members
        for aggregation in protocol["aggregations"]:
            name = tree_name(exponent, str(aggregation))
            prediction = forest.mean if aggregation == "mean" else np.median(forest.members, axis=0)
            predictions[name] = prediction
            retained[f"prediction_{name}"] = prediction
        weight_diagnostics[token] = {
            "minimum": float(np.min(weights)),
            "maximum": float(np.max(weights)),
            "mean": float(np.mean(weights)),
            "standard_deviation": float(np.std(weights)),
            "effective_sample_size": effective_sample_size(weights),
            "seed_prediction_std_mean_ev": float(np.mean(forest.standard_deviation)),
        }

    rows: list[dict[str, Any]] = []
    for model_name, prediction in predictions.items():
        if not np.all(np.isfinite(prediction)):
            raise ValueError(f"non-finite prediction for {element}/{model_name}")
        row: dict[str, Any] = {
            "element": element,
            "model": model_name,
            "train_n": int(train.size),
            "test_n": int(test.size),
            **regression_metrics(targets[test], prediction),
        }
        if model_name.startswith("extra_trees_alpha"):
            prefix, aggregation = model_name.rsplit("_", 1)
            exponent = float(prefix.rsplit("alpha", 1)[1])
            row.update({"frequency_exponent": exponent, "aggregation": aggregation})
        rows.append(row)

    np.savez_compressed(run_dir / f"{element}-predictions.npz", **retained)
    diagnostic = {
        "element": element,
        "train_n": int(train.size),
        "test_n": int(test.size),
        "weight_diagnostics": weight_diagnostics,
        "catboost_prediction_spread_ev": float(np.ptp(catboost_prediction.mean)),
        "catboost_seed_std_mean_ev": float(np.mean(catboost_prediction.standard_deviation)),
    }
    write_json(run_dir / f"{element}-diagnostics.json", diagnostic)
    return rows, diagnostic


def select_candidate(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for exponent_value in protocol["frequency_balance"]["candidate_exponents"]:
        exponent = float(exponent_value)
        for aggregation in protocol["aggregations"]:
            name = tree_name(exponent, str(aggregation))
            matching = [row for row in rows if row["model"] == name]
            if len(matching) != len(protocol["development_elements"]):
                raise ValueError(f"incomplete candidate: {name}")
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
    aggregation_order = {"mean": 0, "median": 1}
    selected = sorted(
        eligible,
        key=lambda item: (float(item["frequency_exponent"]), aggregation_order[str(item["aggregation"])]),
    )[0]
    return {"candidates": candidates, "selected": selected, "tie_band_ev": 0.003}


def development_disposition(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    selection: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    selected = selection["selected"]
    selected_name = str(selected["model"])
    aggregation = str(selected["aggregation"])
    unweighted_name = tree_name(0.0, aggregation)
    selected_rows = {str(row["element"]): row for row in rows if row["model"] == selected_name}
    unweighted_rows = {str(row["element"]): row for row in rows if row["model"] == unweighted_name}
    catboost_rows = {str(row["element"]): row for row in rows if row["model"] == "catboost_ensemble"}
    expected = set(protocol["development_elements"])
    complete = set(selected_rows) == expected and set(unweighted_rows) == expected and set(catboost_rows) == expected
    if not complete:
        raise ValueError("development row coverage is incomplete")
    improvement_unweighted = {
        element: float(unweighted_rows[element]["mae_ev"]) - float(selected_rows[element]["mae_ev"])
        for element in protocol["development_elements"]
    }
    improvement_catboost = {
        element: float(catboost_rows[element]["mae_ev"]) - float(selected_rows[element]["mae_ev"])
        for element in protocol["development_elements"]
    }
    macro_unweighted = float(np.mean(list(improvement_unweighted.values())))
    macro_catboost = float(np.mean(list(improvement_catboost.values())))
    improved_count = int(sum(value > 0.0 for value in improvement_unweighted.values()))
    token = f"{float(selected['frequency_exponent']):.2f}"
    weights_nondegenerate = all(
        float(item["weight_diagnostics"][token]["standard_deviation"]) > 1e-12 for item in diagnostics
    )
    finite = all(
        np.isfinite(float(row[metric]))
        for row in rows
        for metric in ("mae_ev", "rmse_ev", "median_ae_ev", "bias_ev")
    )
    admission = protocol["development_admission_rule"]
    passed = (
        float(selected["frequency_exponent"]) > 0.0
        and macro_unweighted
        > float(admission["minimum_macro_mae_improvement_over_matched_unweighted_extra_trees_ev"])
        and macro_catboost > float(admission["minimum_macro_mae_improvement_over_catboost_ev"])
        and improved_count
        >= int(admission["minimum_improved_elements_over_matched_unweighted_extra_trees"])
        and weights_nondegenerate
        and complete
        and finite
    )
    evidence = {
        "selected_model": selected_name,
        "matched_unweighted_control": unweighted_name,
        "macro_mae_improvement_over_matched_unweighted_extra_trees_ev": macro_unweighted,
        "macro_mae_improvement_over_catboost_ev": macro_catboost,
        "per_element_improvement_over_matched_unweighted_extra_trees_ev": improvement_unweighted,
        "per_element_improvement_over_catboost_ev": improvement_catboost,
        "improved_element_count_over_matched_unweighted_extra_trees": improved_count,
        "weights_nondegenerate_each_element": weights_nondegenerate,
        "all_development_elements_complete": complete,
        "finite_evidence": finite,
        "admission_rule": admission,
        "passed": passed,
    }
    return ("DEVELOPMENT_ADMITTED" if passed else "DEVELOPMENT_NOT_SUPPORTED"), evidence


def run(root: Path, run_id: str, task_type: str) -> dict[str, Any]:
    protocol_path = root / "protocols" / "development-v3.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_DEVELOPMENT":
        raise ValueError("v3 development protocol is not frozen")
    if protocol.get("dataset_sha256") != DATA_SHA256:
        raise ValueError("dataset digest drift")
    if task_type.upper() != str(protocol["catboost"]["task_type"]).upper():
        raise ValueError("execution task type differs from frozen protocol")
    if catboost.__version__ != protocol["catboost"]["version"]:
        raise ValueError(f"CatBoost version drift: {catboost.__version__}")
    if matminer.__version__ != protocol["representation"]["matminer_version"]:
        raise ValueError(f"matminer version drift: {matminer.__version__}")
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.time()
    guard = ResourceGuard(protocol["max_host_rss_mib"], protocol["require_no_rising_pageout"])
    status = "FAILED_RETAINED"
    disposition = "NOT_EVALUATED"
    try:
        guard.check("admission")
        gpu = gpu_inventory()
        if gpu["exit_code"] != 0 or not gpu["stdout"]:
            raise RuntimeError(f"GPU admission failed: {gpu}")
        write_json(run_dir / "gpu-admission.json", gpu)
        formulas, targets = load_data(root)
        observed_counts = {
            element: int(np.sum(formula_element_mask(formulas, element)))
            for element in protocol["development_elements"]
        }
        if observed_counts != protocol["development_element_counts"]:
            raise ValueError(f"development element count drift: {observed_counts}")
        write_json(run_dir / "development-panel.json", {"target_independent_counts": observed_counts})
        matrix = publication_features(formulas)
        write_json(
            run_dir / "feature-schema.json",
            {
                "rows": len(formulas),
                "columns": matrix.values.shape[1],
                "names": list(matrix.names),
                "nan_count": int(np.isnan(matrix.values).sum()),
                "target_independent": True,
            },
        )
        guard.check("features_complete")
        all_rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for element in protocol["development_elements"]:
            rows, diagnostic = evaluate_outer_holdout(
                run_dir, formulas, targets, matrix.values, element, protocol, task_type
            )
            all_rows.extend(rows)
            diagnostics.append(diagnostic)
            guard.check(f"outer_holdout_{element}_complete")
            print(f"SPARKING_V3_OUTER_COMPLETE={element}", flush=True)
        selection = select_candidate(all_rows, protocol)
        disposition, admission = development_disposition(all_rows, diagnostics, selection, protocol)
        result = {
            "schema_version": 3,
            "run_id": run_id,
            "phase": "DEVELOPMENT",
            "status": "SUCCESS",
            "scientific_disposition": disposition,
            "selection": selection,
            "development_admission": admission,
            "rows": all_rows,
            "protected_confirmatory_outcomes_observed": False,
            "official_fold_outcomes_observed": False,
        }
        write_json(run_dir / "results.json", result)
        write_json(
            run_dir / "freeze-candidate.json",
            {
                "eligible_for_confirmatory_freeze": disposition == "DEVELOPMENT_ADMITTED",
                "selected": selection["selected"],
                "source_run_id": run_id,
                "architecture_change_after_freeze": "PROHIBITED",
            },
        )
        guard.check("complete")
        status = "SUCCESS"
        return result
    except BaseException as error:
        write_json(
            run_dir / "failures.json",
            {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        write_json(run_dir / "gpu-postflight.json", gpu_inventory())
        write_json(
            run_dir / "environment.json",
            {
                **environment_record(),
                "catboost": catboost.__version__,
                "matminer": matminer.__version__,
                "numpy": np.__version__,
                "source": package_record(root, protocol_path),
            },
        )
        write_json(
            run_dir / "receipt.json",
            {
                "schema_version": 3,
                "run_id": run_id,
                "status": status,
                "scientific_disposition": disposition,
                "elapsed_seconds": time.time() - started,
                "resource_observations": guard.observations,
            },
        )
        create_manifest(run_dir)
        verify_manifest(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-type", choices=("GPU",), default="GPU")
    args = parser.parse_args()
    result = run(args.root.resolve(), args.run_id, args.task_type)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
