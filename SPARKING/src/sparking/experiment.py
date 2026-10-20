"""Development and confirmatory experiment orchestration."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .chemistry import featurize, feature_names
from .data import DATA_SHA256, element_holdout, load_data, official_folds
from .evidence import (
    ResourceGuard,
    create_manifest,
    environment_record,
    verify_manifest,
    write_json,
)
from .metrics import (
    bootstrap_mean_interval,
    conformal_radius,
    interval_metrics,
    regression_metrics,
)
from .model import SparkingRegressor, direct_hgb_predictions


def _protocol(root: Path, name: str) -> dict[str, Any]:
    path = root / "protocols" / name
    payload = json.loads(path.read_text())
    if payload.get("status") not in {"FROZEN_BEFORE_DEVELOPMENT", "FROZEN_BEFORE_CONFIRMATORY"}:
        raise ValueError(f"protocol is not frozen: {path}")
    if payload.get("dataset_sha256", DATA_SHA256) != DATA_SHA256:
        raise ValueError("protocol dataset digest drift")
    return payload


def _git_record(root: Path) -> dict[str, object]:
    repository = root.parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--", root.name],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {"commit": commit, "project_status": status}


def _fresh_run(root: Path, run_id: str) -> Path:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("run_id may contain only letters, digits, hyphen, and underscore")
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run already exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def _evaluate_holdout(
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    config: dict[str, Any],
    gate_grid: list[tuple[float, float]],
) -> tuple[list[dict[str, object]], SparkingRegressor]:
    x_train, y_train = features[train], targets[train]
    x_test, y_test = features[test], targets[test]
    model = SparkingRegressor.fit(x_train, y_train, config)
    baseline, _, _ = model.components(x_test)
    ungated = model.predict_ungated(x_test)
    direct = direct_hgb_predictions(x_train, y_train, x_test, config)
    median = np.full(y_test.shape, np.median(y_train), dtype=float)
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {
        "median": median,
        "ridge": np.clip(baseline, 0.0, None),
        "direct_hgb": direct,
        "residual_ungated": ungated,
    }
    for model_name, prediction in predictions.items():
        rows.append({"model": model_name, **regression_metrics(y_test, prediction)})
    for quantile, power in gate_grid:
        prediction, gate, distance = model.predict(x_test, quantile, power)
        rows.append(
            {
                "model": "sparking",
                "support_quantile": quantile,
                "power": power,
                "gate_mean": float(gate.mean()),
                "gate_min": float(gate.min()),
                "support_distance_mean": float(distance.mean()),
                "support_threshold": model.threshold(quantile),
                **regression_metrics(y_test, prediction),
            }
        )
    return rows, model


def _failure(run_dir: Path, run_id: str, started: float, guard: ResourceGuard, error: BaseException) -> None:
    write_json(
        run_dir / "failures.json",
        [{"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}],
    )
    write_json(
        run_dir / "receipt.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "FAILED_RETAINED",
            "elapsed_seconds": time.time() - started,
            "resource_observations": guard.observations,
        },
    )
    create_manifest(run_dir)


def run_development(root: Path, run_id: str) -> dict[str, object]:
    protocol = _protocol(root, "development-v1.json")
    run_dir = _fresh_run(root, run_id)
    started = time.time()
    guard = ResourceGuard(protocol["max_rss_mib"], protocol["require_no_rising_pageout"])
    try:
        guard.check("admission")
        formulas, targets = load_data(root)
        guard.check("data_loaded")
        features = featurize(formulas)
        guard.check("features_built")
        config = protocol["residual_model"]
        grid = [
            (float(quantile), float(power))
            for quantile, power in itertools.product(
                protocol["gate_grid"]["support_quantile"], protocol["gate_grid"]["power"]
            )
        ]
        all_rows: list[dict[str, object]] = []
        for element in protocol["development_elements"]:
            train, test = element_holdout(formulas, element)
            rows, model = _evaluate_holdout(features, targets, train, test, config, grid)
            for row in rows:
                row["element"] = element
                row["train_n"] = int(train.size)
                row["test_n"] = int(test.size)
                all_rows.append(row)
            joblib.dump(
                {
                    "schema_version": 1,
                    "element": element,
                    "model": model,
                    "feature_names": feature_names(),
                },
                run_dir / f"development-{element}.joblib",
                compress=3,
            )
            guard.check(f"holdout_{element}")

        candidates: list[dict[str, float]] = []
        for quantile, power in grid:
            matching = [
                row for row in all_rows
                if row["model"] == "sparking"
                and row["support_quantile"] == quantile
                and row["power"] == power
            ]
            candidates.append(
                {
                    "support_quantile": quantile,
                    "power": power,
                    "macro_mae_ev": float(np.mean([float(row["mae_ev"]) for row in matching])),
                }
            )
        minimum = min(candidate["macro_mae_ev"] for candidate in candidates)
        eligible = [candidate for candidate in candidates if candidate["macro_mae_ev"] <= minimum + 0.002]
        selected = sorted(eligible, key=lambda item: (item["power"], -item["support_quantile"]))[0]
        controls = {
            name: float(np.mean([float(row["mae_ev"]) for row in all_rows if row["model"] == name]))
            for name in ("median", "ridge", "direct_hgb", "residual_ungated")
        }
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "phase": "DEVELOPMENT",
            "status": "SUCCESS",
            "selected_gate": selected,
            "candidate_summary": candidates,
            "control_macro_mae_ev": controls,
            "rows": all_rows,
            "protected_confirmatory_outcomes_observed": False,
        }
        write_json(run_dir / "results.json", result)
        freeze_candidate = {
            "schema_version": 1,
            "source_run_id": run_id,
            "support_quantile": selected["support_quantile"],
            "power": selected["power"],
            "selection_rule": protocol["selection_rule"],
            "architecture_change_after_freeze": "PROHIBITED",
        }
        write_json(run_dir / "freeze_candidate.json", freeze_candidate)
        write_json(run_dir / "environment.json", {**environment_record(), "git": _git_record(root)})
        guard.check("complete")
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "SUCCESS",
            "elapsed_seconds": time.time() - started,
            "resource_observations": guard.observations,
        }
        write_json(run_dir / "receipt.json", receipt)
        create_manifest(run_dir)
        verify_manifest(run_dir)
        return result
    except BaseException as error:
        _failure(run_dir, run_id, started, guard, error)
        raise


def _one_sided_sign_permutation(values: np.ndarray) -> float:
    observed = float(np.mean(values))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        statistics.append(float(np.mean(values * np.asarray(signs))))
    return float((sum(value >= observed - 1e-15 for value in statistics)) / len(statistics))


def run_confirmatory(root: Path, run_id: str) -> dict[str, object]:
    protocol = _protocol(root, "confirmatory-v1.json")
    run_dir = _fresh_run(root, run_id)
    started = time.time()
    guard = ResourceGuard(protocol["max_rss_mib"], protocol["require_no_rising_pageout"])
    try:
        guard.check("admission")
        formulas, targets = load_data(root)
        features = featurize(formulas)
        guard.check("features_built")
        config = protocol["residual_model"]
        quantile = float(protocol["selected_gate"]["support_quantile"])
        power = float(protocol["selected_gate"]["power"])
        holdout_rows: list[dict[str, object]] = []
        for element in protocol["confirmatory_elements"]:
            train, test = element_holdout(formulas, element)
            rows, _ = _evaluate_holdout(features, targets, train, test, config, [(quantile, power)])
            for row in rows:
                row["element"] = element
                holdout_rows.append(row)
            guard.check(f"confirmatory_holdout_{element}")

        prediction_rows: list[dict[str, object]] = []
        interval_rows: list[dict[str, object]] = []
        for fold, (train, test) in enumerate(official_folds(len(formulas))):
            model = SparkingRegressor.fit(features[train], targets[train], config)
            sparking, gate, distance = model.predict(features[test], quantile, power)
            ungated = model.predict_ungated(features[test])
            ridge, _, _ = model.components(features[test])
            direct = direct_hgb_predictions(features[train], targets[train], features[test], config)
            for local, index in enumerate(test):
                prediction_rows.append(
                    {
                        "row_index": int(index),
                        "formula": formulas[int(index)],
                        "target_ev": float(targets[int(index)]),
                        "fold": fold,
                        "sparking_ev": float(sparking[local]),
                        "ungated_ev": float(ungated[local]),
                        "ridge_ev": float(max(0.0, ridge[local])),
                        "direct_hgb_ev": float(direct[local]),
                        "gate": float(gate[local]),
                        "support_distance": float(distance[local]),
                    }
                )

            proper, calibration = train_test_split(
                train, test_size=0.2, random_state=20261020 + fold, shuffle=True
            )
            uq_model = SparkingRegressor.fit(features[proper], targets[proper], config)
            calibration_prediction, _, _ = uq_model.predict(features[calibration], quantile, power)
            radius = conformal_radius(np.abs(targets[calibration] - calibration_prediction), 0.9)
            uq_prediction, _, _ = uq_model.predict(features[test], quantile, power)
            interval_rows.append(
                {
                    "fold": fold,
                    "calibration_n": int(calibration.size),
                    "radius_ev": radius,
                    **interval_metrics(
                        targets[test],
                        np.clip(uq_prediction - radius, 0.0, None),
                        uq_prediction + radius,
                        0.9,
                    ),
                }
            )
            guard.check(f"official_fold_{fold}")

        prediction_rows.sort(key=lambda row: int(row["row_index"]))
        predictions_path = run_dir / "predictions.csv"
        with predictions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
        truth = np.asarray([row["target_ev"] for row in prediction_rows], dtype=float)
        official = {
            name: regression_metrics(truth, np.asarray([row[f"{name}_ev"] for row in prediction_rows], dtype=float))
            for name in ("sparking", "ungated", "ridge", "direct_hgb")
        }
        deltas = []
        element_results = []
        for element in protocol["confirmatory_elements"]:
            sparking_row = next(row for row in holdout_rows if row["element"] == element and row["model"] == "sparking")
            ungated_row = next(row for row in holdout_rows if row["element"] == element and row["model"] == "residual_ungated")
            delta = float(ungated_row["mae_ev"]) - float(sparking_row["mae_ev"])
            deltas.append(delta)
            element_results.append(
                {
                    "element": element,
                    "sparking_mae_ev": sparking_row["mae_ev"],
                    "ungated_mae_ev": ungated_row["mae_ev"],
                    "improvement_ev": delta,
                }
            )
        delta_array = np.asarray(deltas, dtype=float)
        macro_delta = float(delta_array.mean())
        improved = int(np.sum(delta_array > 0.0))
        official_degradation = float(official["sparking"]["mae_ev"] - official["ungated"]["mae_ev"])
        rule = protocol["success_rule"]
        supported = (
            macro_delta > float(rule["macro_mae_improvement_over_ungated_ev"])
            and improved >= int(rule["minimum_improved_elements"])
            and official_degradation <= float(rule["maximum_official_mae_degradation_ev"])
        )
        if supported:
            disposition = "SUPPORTED"
        elif macro_delta > 0.0 and improved >= 3:
            disposition = "MIXED"
        else:
            disposition = "NEGATIVE"
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "phase": "CONFIRMATORY",
            "status": "SUCCESS",
            "scientific_disposition": disposition,
            "selected_gate": {"support_quantile": quantile, "power": power},
            "element_results": element_results,
            "macro_improvement": bootstrap_mean_interval(delta_array),
            "one_sided_sign_permutation_p": _one_sided_sign_permutation(delta_array),
            "improved_element_count": improved,
            "official_metrics": official,
            "official_mae_degradation_ev": official_degradation,
            "interval_90_by_fold": interval_rows,
            "success_rule": rule,
        }
        write_json(run_dir / "results.json", result)
        write_json(run_dir / "holdout_metrics.json", holdout_rows)
        write_json(run_dir / "environment.json", {**environment_record(), "git": _git_record(root)})
        guard.check("complete")
        write_json(
            run_dir / "receipt.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": "SUCCESS",
                "scientific_disposition": disposition,
                "elapsed_seconds": time.time() - started,
                "resource_observations": guard.observations,
            },
        )
        create_manifest(run_dir)
        verify_manifest(run_dir)
        return result
    except BaseException as error:
        _failure(run_dir, run_id, started, guard, error)
        raise

