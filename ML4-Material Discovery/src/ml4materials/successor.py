"""Versioned successor experiment runners that never mutate frozen v1 runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from .data import (
    load_data_contract,
    load_json,
    load_official_splits,
    load_records,
    sha256_file,
    write_json,
)
from .evaluation import interval_metrics, regression_metrics
from .experiment import create_artifact_manifest
from .features import CompositionFeaturizer
from .guards import ResourceGuard
from .models import (
    fit_quantile_interval_models,
    load_quantile_checkpoint,
    make_quantile_gradient_boosting,
    make_ridge,
    quantile_interval_predictions,
    save_quantile_checkpoint,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _indices(ids: list[str], index_by_id: dict[str, int]) -> np.ndarray:
    return np.asarray([index_by_id[identifier] for identifier in ids], dtype=int)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_quantile_study(root: Path, run_id: str, config_path: Path) -> Path:
    """Execute the v2 quantile successor under the same frozen resource guard.

    This runner is intentionally independent of ``run_study``: it neither
    overwrites v1 receipts nor claims direct-quantile results are v1 results.
    """

    from .plotting import calibration_plot, parity_plot

    root = root.resolve()
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    config = load_json(config_path)
    if config.get("parent_study") != "development-v1":
        raise ValueError("quantile successor must declare development-v1 as parent")
    run_dir.mkdir(parents=True)
    started = _utc_now()
    started_monotonic = time.monotonic()
    failures: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    try:
        contract = load_data_contract(root, str(config["raw_data_contract"]))
        records = load_records(root, contract)
        splits = load_official_splits(root, contract, records)
        guard = ResourceGuard(
            max_allowed_rss_mib=float(config["max_rss_mib"]),
            require_no_pageout_increase=bool(config["require_no_pageout_increase"]),
        )
        guard.check("data_validation")
        formulas = [record.composition for record in records]
        targets = np.asarray([record.gap_ev for record in records], dtype=float)
        index_by_id = {record.material_id: index for index, record in enumerate(records)}
        point_predictions = np.full(len(records), np.nan)
        ridge_predictions = np.full(len(records), np.nan)
        fold_names = np.full(len(records), "", dtype=object)
        interval_bounds = {
            float(coverage): (np.full(len(records), np.nan), np.full(len(records), np.nan))
            for coverage in config["interval_coverages"]
        }
        fold_rows: list[dict[str, Any]] = []

        for fold_name in config["official_folds"]:
            started_fold = time.monotonic()
            train_indices = _indices(splits[fold_name]["train"], index_by_id)
            test_indices = _indices(splits[fold_name]["test"], index_by_id)
            featurizer = CompositionFeaturizer().fit([formulas[index] for index in train_indices])
            train_features = featurizer.transform([formulas[index] for index in train_indices])
            test_features = featurizer.transform([formulas[index] for index in test_indices])
            point_model = make_quantile_gradient_boosting(0.5, config["point_model"])
            point_model.fit(train_features, targets[train_indices])
            point = np.asarray(point_model.predict(test_features), dtype=float)
            interval_models: dict[str, dict[str, Any]] = {}
            for coverage in config["interval_coverages"]:
                lower_model, upper_model = fit_quantile_interval_models(
                    train_features,
                    targets[train_indices],
                    float(coverage),
                    config["quantile_model"],
                )
                lower, upper = quantile_interval_predictions(lower_model, upper_model, test_features)
                interval_bounds[float(coverage)][0][test_indices] = lower
                interval_bounds[float(coverage)][1][test_indices] = upper
                interval_models[str(coverage)] = {"lower": lower_model, "upper": upper_model}
            ridge = make_ridge(config.get("ridge_alphas", [0.001, 0.01, 0.1, 1, 10, 100]))
            ridge.fit(train_features, targets[train_indices])
            ridge_prediction = np.asarray(ridge.predict(test_features), dtype=float)
            checkpoint_path = run_dir / "models" / f"{fold_name}.joblib"
            save_quantile_checkpoint(
                checkpoint_path,
                {
                    "fold": fold_name,
                    "point_model": point_model,
                    "interval_models": interval_models,
                    "feature_names": list(featurizer.feature_names),
                    "train_element_counts": featurizer.train_element_counts,
                    "train_ids": splits[fold_name]["train"],
                },
            )
            reloaded = load_quantile_checkpoint(checkpoint_path)
            reloaded_point = np.asarray(reloaded["point_model"].predict(test_features), dtype=float)
            if not np.array_equal(point, reloaded_point):
                raise RuntimeError(f"quantile checkpoint reload changed {fold_name} predictions")
            point_predictions[test_indices] = point
            ridge_predictions[test_indices] = ridge_prediction
            fold_names[test_indices] = fold_name
            fold_rows.append(
                {
                    "fold": fold_name,
                    "train_n": int(len(train_indices)),
                    "test_n": int(len(test_indices)),
                    "quantile_point_mae_ev": regression_metrics(targets[test_indices], point)["mae_ev"],
                    "ridge_mae_ev": regression_metrics(targets[test_indices], ridge_prediction)["mae_ev"],
                    "duration_seconds": time.monotonic() - started_fold,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            )
            events.append({"event": "official_fold_complete", "fold": fold_name, "utc": _utc_now()})
            guard.check(f"official_{fold_name}")

        if not (np.all(np.isfinite(point_predictions)) and np.all(np.isfinite(ridge_predictions))):
            raise RuntimeError("incomplete out-of-fold successor predictions")
        calibration = [
            interval_metrics(targets, lower, upper, coverage)
            for coverage, (lower, upper) in sorted(interval_bounds.items())
        ]
        prediction_rows = [
            {
                "material_id": record.material_id,
                "composition": record.composition,
                "canonical_composition": record.canonical_composition,
                "fold": str(fold_names[index]),
                "target_gap_ev": float(targets[index]),
                "quantile_point_prediction_ev": float(point_predictions[index]),
                "ridge_prediction_ev": float(ridge_predictions[index]),
                **{
                    f"interval_{int(coverage * 100)}_lower_ev": float(lower[index])
                    for coverage, (lower, _) in interval_bounds.items()
                },
                **{
                    f"interval_{int(coverage * 100)}_upper_ev": float(upper[index])
                    for coverage, (_, upper) in interval_bounds.items()
                },
            }
            for index, record in enumerate(records)
        ]
        results = {
            "schema_version": 1,
            "status": "EXECUTED_SUCCESSOR_NOT_INDEPENDENTLY_VALIDATED",
            "study_version": config["study_version"],
            "parent_study": config["parent_study"],
            "metrics": {
                "quantile_point": regression_metrics(targets, point_predictions),
                "ridge": regression_metrics(targets, ridge_predictions),
            },
            "calibration": calibration,
            "folds": fold_rows,
        }
        write_json(run_dir / "results.json", results)
        _write_csv(run_dir / "tables/folds.csv", fold_rows)
        _write_csv(run_dir / "tables/calibration.csv", calibration)
        _write_csv(run_dir / "predictions/oof_predictions.csv", prediction_rows)
        parity_plot(targets, point_predictions, ridge_predictions, run_dir / "figures/parity.png")
        calibration_plot(calibration, run_dir / "figures/calibration.png")
        guard.check("figure_generation")
        receipt = {
            "run_id": run_id,
            "status": "COMPLETED",
            "started_utc": started,
            "finished_utc": _utc_now(),
            "duration_seconds": time.monotonic() - started_monotonic,
            "inputs": {
                "dataset_sha256": contract["source_sha256"],
                "split_sha256": contract["split_source_sha256"],
            },
            "resources": guard.receipt(),
        }
    except Exception as exc:
        failures.append(
            {"utc": _utc_now(), "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        )
        receipt = {
            "run_id": run_id,
            "status": "FAILED_RETAINED",
            "started_utc": started,
            "finished_utc": _utc_now(),
            "duration_seconds": time.monotonic() - started_monotonic,
        }
        if "guard" in locals():
            receipt["resources"] = guard.receipt()
    write_json(run_dir / "events.json", events)
    write_json(run_dir / "failures.json", failures)
    write_json(run_dir / "receipt.json", receipt)
    create_artifact_manifest(run_dir)
    if receipt["status"] != "COMPLETED":
        raise RuntimeError(f"quantile successor failed; retained receipt: {run_dir}")
    return run_dir
