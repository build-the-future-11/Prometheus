#!/usr/bin/env python3
"""Execute the frozen SPARKING v2 development study and retain raw evidence."""

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
    RISK_FEATURE_NAMES,
    RiskContext,
    catboost_ensemble_predictions,
    convex_prediction,
    extra_trees_ensemble_predictions,
    fit_harm_model,
    formula_element_mask,
    make_publication_ridge,
    publication_features,
    risk_gate_weights,
    weighted_spearman,
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
) -> tuple[np.ndarray, Any]:
    model = make_publication_ridge()
    model.fit(features[train], targets[train])
    prediction = np.clip(np.asarray(model.predict(features[test]), dtype=float), 0.0, None)
    return prediction, model


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


def pseudo_calibration(
    formulas: list[str],
    features: np.ndarray,
    base_features: np.ndarray,
    targets: np.ndarray,
    outer_train: np.ndarray,
    protocol: dict[str, Any],
    task_type: str,
) -> tuple[Any, dict[str, np.ndarray], list[dict[str, Any]]]:
    risk_rows: list[np.ndarray] = []
    harm_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    element_rows: list[np.ndarray] = []
    row_index_rows: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    seeds = [int(seed) for seed in protocol["catboost"]["seeds"]]
    minimum = int(protocol["minimum_pseudo_holdout_rows"])
    for element_index, element in enumerate(protocol["pseudo_exclusion_elements"]):
        mask = formula_element_mask(formulas, element)
        pseudo_test = outer_train[mask[outer_train]]
        pseudo_train = outer_train[~mask[outer_train]]
        if pseudo_test.size < minimum:
            raise ValueError(f"pseudo exclusion {element} has only {pseudo_test.size} rows")
        ridge, _ = fit_ridge_predict(features, targets, pseudo_train, pseudo_test)
        ensemble = catboost_ensemble_predictions(
            features[pseudo_train],
            targets[pseudo_train],
            features[pseudo_test],
            protocol["catboost"],
            seeds,
            task_type,
        )
        context = RiskContext.fit(features[pseudo_train], base_features[pseudo_train])
        risk = context.transform(
            features[pseudo_test],
            base_features[pseudo_test],
            ensemble.mean,
            ensemble.standard_deviation,
            ridge,
        )
        harm = np.abs(targets[pseudo_test] - ensemble.mean) - np.abs(targets[pseudo_test] - ridge)
        risk_rows.append(risk)
        harm_rows.append(harm)
        weight_rows.append(np.full(pseudo_test.size, 1.0 / pseudo_test.size, dtype=float))
        element_rows.append(np.full(pseudo_test.size, element_index, dtype=int))
        row_index_rows.append(pseudo_test.astype(int))
        diagnostics.append(
            {
                "element": element,
                "train_n": int(pseudo_train.size),
                "test_n": int(pseudo_test.size),
                "direct_metrics": regression_metrics(targets[pseudo_test], ensemble.mean),
                "ridge_metrics": regression_metrics(targets[pseudo_test], ridge),
                "direct_harm_mean_ev": float(np.mean(harm)),
                "direct_harm_positive_fraction": float(np.mean(harm > 0.0)),
                "ensemble_prediction_std_mean_ev": float(np.mean(ensemble.standard_deviation)),
            }
        )
    risk_matrix = np.vstack(risk_rows)
    harms = np.concatenate(harm_rows)
    weights = np.concatenate(weight_rows)
    weights *= len(protocol["pseudo_exclusion_elements"]) / float(np.sum(weights))
    model = fit_harm_model(risk_matrix, harms, weights, protocol["harm_model"])
    predicted = np.asarray(model.predict(risk_matrix), dtype=float)
    arrays = {
        "risk_features": risk_matrix,
        "direct_harm_ev": harms,
        "sample_weight": weights,
        "pseudo_element_index": np.concatenate(element_rows),
        "row_index": np.concatenate(row_index_rows),
        "fitted_harm_ev": predicted,
    }
    return model, arrays, diagnostics


def evaluate_outer_holdout(
    root: Path,
    run_dir: Path,
    formulas: list[str],
    targets: np.ndarray,
    features: np.ndarray,
    base_features: np.ndarray,
    element: str,
    protocol: dict[str, Any],
    task_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mask = formula_element_mask(formulas, element)
    train = np.flatnonzero(~mask)
    test = np.flatnonzero(mask)
    harm_model, pseudo_arrays, pseudo_diagnostics = pseudo_calibration(
        formulas, features, base_features, targets, train, protocol, task_type
    )
    np.savez_compressed(run_dir / f"{element}-pseudo-calibration.npz", **pseudo_arrays)

    ridge, _ = fit_ridge_predict(features, targets, train, test)
    seeds = [int(seed) for seed in protocol["catboost"]["seeds"]]
    catboost_prediction = catboost_ensemble_predictions(
        features[train], targets[train], features[test], protocol["catboost"], seeds, task_type
    )
    extra_trees = extra_trees_ensemble_predictions(
        features[train],
        targets[train],
        features[test],
        protocol["extra_trees"],
        protocol["extra_trees"]["seeds"],
    )
    hgb = fit_hgb_squared_predict(features, targets, train, test, protocol["corrected_hgb"])
    risk_context = RiskContext.fit(features[train], base_features[train])
    risk_features = risk_context.transform(
        features[test],
        base_features[test],
        catboost_prediction.mean,
        catboost_prediction.standard_deviation,
        ridge,
    )
    predicted_harm = np.asarray(harm_model.predict(risk_features), dtype=float)
    observed_harm = np.abs(targets[test] - catboost_prediction.mean) - np.abs(targets[test] - ridge)
    median = np.full(test.size, float(np.median(targets[train])), dtype=float)
    predictions: dict[str, np.ndarray] = {
        "median": median,
        "ridge": ridge,
        "direct_hgb_squared": hgb,
        "extra_trees_ensemble": extra_trees.mean,
        "catboost_single_seed": catboost_prediction.members[0],
        "catboost_ensemble": catboost_prediction.mean,
    }
    for weight in (0.25, 0.5, 0.75):
        predictions[f"constant_blend_{weight:.2f}"] = convex_prediction(ridge, catboost_prediction.mean, weight)
    gate_arrays: dict[str, np.ndarray] = {}
    for temperature in protocol["gate_temperatures_ev"]:
        value = float(temperature)
        weights = risk_gate_weights(predicted_harm, value)
        name = f"risk_gate_t{value:.2f}"
        predictions[name] = convex_prediction(ridge, catboost_prediction.mean, weights)
        gate_arrays[name] = weights

    rows: list[dict[str, Any]] = []
    for model_name, prediction in predictions.items():
        row: dict[str, Any] = {
            "element": element,
            "model": model_name,
            "train_n": int(train.size),
            "test_n": int(test.size),
            **regression_metrics(targets[test], prediction),
        }
        if model_name.startswith("risk_gate_t"):
            weights = gate_arrays[model_name]
            row.update(
                {
                    "temperature_ev": float(model_name.rsplit("t", 1)[1]),
                    "gate_mean": float(np.mean(weights)),
                    "gate_min": float(np.min(weights)),
                    "gate_active_fraction": float(np.mean(weights < 1.0 - 1e-12)),
                    "mean_abs_change_from_catboost_ev": float(
                        np.mean(np.abs(prediction - catboost_prediction.mean))
                    ),
                }
            )
        rows.append(row)

    np.savez_compressed(
        run_dir / f"{element}-predictions.npz",
        row_index=test,
        target_ev=targets[test],
        ridge_ev=ridge,
        hgb_squared_ev=hgb,
        extra_trees_members_ev=extra_trees.members,
        catboost_members_ev=catboost_prediction.members,
        catboost_mean_ev=catboost_prediction.mean,
        catboost_std_ev=catboost_prediction.standard_deviation,
        risk_features=risk_features,
        predicted_direct_harm_ev=predicted_harm,
        observed_direct_harm_ev=observed_harm,
        **{f"prediction_{name}": values for name, values in predictions.items()},
        **{f"weight_{name}": values for name, values in gate_arrays.items()},
    )
    diagnostic = {
        "element": element,
        "pseudo_exclusions": pseudo_diagnostics,
        "pseudo_calibration_rows": int(pseudo_arrays["direct_harm_ev"].size),
        "pseudo_harm_fit_spearman": weighted_spearman(
            pseudo_arrays["fitted_harm_ev"], pseudo_arrays["direct_harm_ev"]
        ),
        "outer_risk_harm_spearman_descriptive": weighted_spearman(predicted_harm, observed_harm),
        "catboost_prediction_spread_ev": float(np.ptp(catboost_prediction.mean)),
        "catboost_seed_std_mean_ev": float(np.mean(catboost_prediction.standard_deviation)),
        "extra_trees_seed_std_mean_ev": float(np.mean(extra_trees.standard_deviation)),
        "risk_feature_names": list(RISK_FEATURE_NAMES),
    }
    write_json(run_dir / f"{element}-diagnostics.json", diagnostic)
    return rows, diagnostic


def select_candidate(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for temperature in protocol["gate_temperatures_ev"]:
        name = f"risk_gate_t{float(temperature):.2f}"
        matching = [row for row in rows if row["model"] == name]
        if len(matching) != len(protocol["development_elements"]):
            raise ValueError(f"incomplete candidate {name}")
        candidates.append(
            {
                "model": name,
                "temperature_ev": float(temperature),
                "macro_mae_ev": float(np.mean([float(row["mae_ev"]) for row in matching])),
            }
        )
    minimum = min(float(candidate["macro_mae_ev"]) for candidate in candidates)
    eligible = [candidate for candidate in candidates if float(candidate["macro_mae_ev"]) <= minimum + 0.005]
    selected = sorted(eligible, key=lambda item: -float(item["temperature_ev"]))[0]
    return {"candidates": candidates, "selected": selected}


def development_disposition(
    rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]], selection: dict[str, Any], protocol: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    selected_name = str(selection["selected"]["model"])
    selected_rows = [row for row in rows if row["model"] == selected_name]
    direct_rows = [row for row in rows if row["model"] == "catboost_ensemble"]
    direct_by_element = {str(row["element"]): row for row in direct_rows}
    improvements = {
        str(row["element"]): float(direct_by_element[str(row["element"])]["mae_ev"]) - float(row["mae_ev"])
        for row in selected_rows
    }
    macro_improvement = float(np.mean(list(improvements.values())))
    improved = int(sum(value > 0.0 for value in improvements.values()))
    admission = protocol["development_admission_rule"]
    activity_ok = all(
        float(admission["minimum_active_fraction_each_element"])
        <= float(row["gate_active_fraction"])
        <= float(admission["maximum_active_fraction_each_element"])
        for row in selected_rows
    )
    nonconstant = all(float(item["catboost_prediction_spread_ev"]) > 1e-8 for item in diagnostics)
    complete_pseudo = all(
        len(item["pseudo_exclusions"]) == len(protocol["pseudo_exclusion_elements"]) for item in diagnostics
    )
    passed = (
        macro_improvement > float(admission["minimum_macro_mae_improvement_over_catboost_ev"])
        and improved >= int(admission["minimum_improved_elements"])
        and activity_ok
        and nonconstant
        and complete_pseudo
    )
    evidence = {
        "macro_mae_improvement_over_catboost_ev": macro_improvement,
        "per_element_improvement_ev": improvements,
        "improved_element_count": improved,
        "gate_activity_within_bounds": activity_ok,
        "catboost_predictions_nonconstant": nonconstant,
        "all_pseudo_exclusions_complete": complete_pseudo,
        "admission_rule": admission,
        "passed": passed,
    }
    return ("DEVELOPMENT_ADMITTED" if passed else "DEVELOPMENT_NOT_SUPPORTED"), evidence


def run(root: Path, run_id: str, task_type: str) -> dict[str, Any]:
    protocol_path = root / "protocols" / "development-v2.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_DEVELOPMENT":
        raise ValueError("v2 development protocol is not frozen")
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
                root,
                run_dir,
                formulas,
                targets,
                matrix.values,
                matrix.base_values,
                element,
                protocol,
                task_type,
            )
            all_rows.extend(rows)
            diagnostics.append(diagnostic)
            guard.check(f"outer_holdout_{element}_complete")
            print(f"SPARKING_V2_OUTER_COMPLETE={element}", flush=True)
        selection = select_candidate(all_rows, protocol)
        disposition, admission = development_disposition(all_rows, diagnostics, selection, protocol)
        result = {
            "schema_version": 2,
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
                "schema_version": 2,
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
