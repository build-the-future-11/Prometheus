"""End-to-end frozen development experiment with durable receipts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

from .chemistry import element_fractions
from .data import (
    MaterialRecord,
    duplicate_audit,
    load_data_contract,
    load_json,
    load_official_splits,
    load_records,
    prepare_manifests,
    sha256_file,
    split_leakage_audit,
    validate_group_isolation,
    write_json,
)
from .evaluation import (
    abstention_metrics,
    regression_metrics,
    subgroup_metrics,
)
from .features import CompositionFeaturizer
from .guards import ResourceGuard
from .models import (
    ensemble_predictions,
    fit_ensemble,
    load_checkpoint,
    make_extra_trees,
    make_ridge,
    save_checkpoint,
)
from .ranking import evaluate_ranking
from .report import render_manuscript, render_results
from .uncertainty import NormalizedConformalCalibrator, calibration_partition


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(encoded)
    temporary.replace(path)


def _indices(ids: list[str], index_by_id: dict[str, int]) -> np.ndarray:
    return np.asarray([index_by_id[identifier] for identifier in ids], dtype=int)


def _git_value(root: Path, arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _fold_checkpoint_prediction(
    checkpoint: dict[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction, raw_sigma, _ = ensemble_predictions(checkpoint["models"], features)
    calibrator = NormalizedConformalCalibrator.from_dict(checkpoint["calibrator"])
    return prediction, calibrator.calibrated_sigma(raw_sigma)


def _pooled_calibration(
    targets: np.ndarray,
    interval_bounds: dict[float, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for coverage, (lower, upper) in sorted(interval_bounds.items()):
        observed = float(np.mean((targets >= lower) & (targets <= upper)))
        rows.append(
            {
                "nominal_coverage": coverage,
                "observed_coverage": observed,
                "coverage_error": observed - coverage,
                "mean_interval_width_ev": float(np.mean(upper - lower)),
            }
        )
    return rows


def _candidate_pool(
    records: list[MaterialRecord],
    fold_names: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.canonical_composition, []).append(index)
    metadata: list[dict[str, Any]] = []
    pooled_target: list[float] = []
    pooled_prediction: list[float] = []
    pooled_uncertainty: list[float] = []
    for canonical, indices in sorted(groups.items()):
        index_array = np.asarray(indices, dtype=int)
        group_prediction = predictions[index_array]
        group_uncertainty = uncertainty[index_array]
        combined_variance = float(
            np.mean(group_uncertainty**2) + np.var(group_prediction, ddof=0)
        )
        source_targets = targets[index_array]
        metadata.append(
            {
                "canonical_composition": canonical,
                "representative_formula": records[indices[0]].composition,
                "source_ids": [records[index].material_id for index in indices],
                "source_formulas": [records[index].composition for index in indices],
                "source_folds": [str(fold_names[index]) for index in indices],
                "source_target_values_ev": [float(value) for value in source_targets],
                "source_target_span_ev": float(source_targets.max() - source_targets.min()),
                "source_row_count": len(indices),
            }
        )
        pooled_target.append(float(np.mean(source_targets)))
        pooled_prediction.append(float(np.mean(group_prediction)))
        pooled_uncertainty.append(float(np.sqrt(combined_variance)))
    return (
        metadata,
        np.asarray(pooled_target),
        np.asarray(pooled_prediction),
        np.asarray(pooled_uncertainty),
    )


def _generalization_study(
    records: list[MaterialRecord],
    config: dict[str, Any],
    group_by_id: dict[str, str],
    guard: ResourceGuard,
) -> list[dict[str, Any]]:
    formulas = [record.composition for record in records]
    targets = np.asarray([record.gap_ev for record in records], dtype=float)
    index_by_id = {record.material_id: index for index, record in enumerate(records)}
    results: list[dict[str, Any]] = []
    for element in config["element_holdouts"]:
        test_indices = np.asarray(
            [index for index, formula in enumerate(formulas) if element in element_fractions(formula)],
            dtype=int,
        )
        test_index_set = set(test_indices.tolist())
        train_indices = np.asarray(
            [index for index in range(len(records)) if index not in test_index_set],
            dtype=int,
        )
        if len(test_indices) < int(config["minimum_holdout_rows"]):
            raise ValueError(
                f"element holdout {element} has {len(test_indices)} rows, below frozen minimum"
            )
        train_ids = [records[index].material_id for index in train_indices]
        test_ids = [records[index].material_id for index in test_indices]
        validate_group_isolation(train_ids, test_ids, records)
        fit_ids, calibration_ids = calibration_partition(
            train_ids,
            float(config["calibration_fraction"]),
            int(config["calibration_seed"]),
            group_by_id,
        )
        fit_indices = _indices(fit_ids, index_by_id)
        calibration_indices = _indices(calibration_ids, index_by_id)
        featurizer = CompositionFeaturizer().fit([formulas[index] for index in fit_indices])
        fit_features = featurizer.transform([formulas[index] for index in fit_indices])
        calibration_features = featurizer.transform(
            [formulas[index] for index in calibration_indices]
        )
        test_features = featurizer.transform([formulas[index] for index in test_indices])
        models = fit_ensemble(
            fit_features,
            targets[fit_indices],
            config["model_seeds"],
            config["extra_trees"],
        )
        calibration_prediction, calibration_raw_sigma, _ = ensemble_predictions(
            models, calibration_features
        )
        calibrator = NormalizedConformalCalibrator.fit(
            targets[calibration_indices], calibration_prediction, calibration_raw_sigma
        )
        proposed_prediction, raw_sigma, seed_predictions = ensemble_predictions(
            models, test_features
        )
        lower_90, upper_90, _ = calibrator.interval(proposed_prediction, raw_sigma, 0.9)

        ridge_featurizer = CompositionFeaturizer().fit(
            [formulas[index] for index in train_indices]
        )
        ridge_train = ridge_featurizer.transform(
            [formulas[index] for index in train_indices]
        )
        ridge_test = ridge_featurizer.transform([formulas[index] for index in test_indices])
        ridge = make_ridge(config["ridge_alphas"])
        ridge.fit(ridge_train, targets[train_indices])
        ridge_prediction = ridge.predict(ridge_test)
        results.append(
            {
                "held_out_element": element,
                "train_n": int(len(train_indices)),
                "proper_train_n": int(len(fit_indices)),
                "calibration_n": int(len(calibration_indices)),
                "test_n": int(len(test_indices)),
                "proposed": regression_metrics(targets[test_indices], proposed_prediction),
                "ridge": regression_metrics(targets[test_indices], ridge_prediction),
                "trivial": regression_metrics(
                    targets[test_indices],
                    np.full(len(test_indices), np.median(targets[train_indices])),
                ),
                "coverage_90": float(
                    np.mean(
                        (targets[test_indices] >= lower_90)
                        & (targets[test_indices] <= upper_90)
                    )
                ),
                "mean_interval_width_90_ev": float(np.mean(upper_90 - lower_90)),
                "per_seed_mae_ev": [
                    float(np.mean(np.abs(targets[test_indices] - seed_predictions[:, column])))
                    for column in range(seed_predictions.shape[1])
                ],
            }
        )
        guard.check(f"element_holdout_{element}")
    return results


def _permutation_control(
    records: list[MaterialRecord],
    splits: dict[str, dict[str, list[str]]],
    config: dict[str, Any],
    group_by_id: dict[str, str],
    unpermuted_mae: float,
) -> dict[str, Any]:
    formulas = [record.composition for record in records]
    targets = np.asarray([record.gap_ev for record in records], dtype=float)
    index_by_id = {record.material_id: index for index, record in enumerate(records)}
    train_ids = splits["fold_0"]["train"]
    test_indices = _indices(splits["fold_0"]["test"], index_by_id)
    fit_ids, _ = calibration_partition(
        train_ids,
        float(config["calibration_fraction"]),
        int(config["calibration_seed"]),
        group_by_id,
    )
    fit_indices = _indices(fit_ids, index_by_id)
    featurizer = CompositionFeaturizer().fit([formulas[index] for index in fit_indices])
    fit_features = featurizer.transform([formulas[index] for index in fit_indices])
    test_features = featurizer.transform([formulas[index] for index in test_indices])
    maes: list[float] = []
    for seed in config["permutation_seeds"]:
        rng = np.random.default_rng(int(seed))
        permuted = rng.permutation(targets[fit_indices])
        model = make_extra_trees(int(seed), config["extra_trees"])
        model.fit(fit_features, permuted)
        prediction = model.predict(test_features)
        maes.append(float(np.mean(np.abs(targets[test_indices] - prediction))))
    mean_mae = float(np.mean(maes))
    return {
        "fold": "fold_0",
        "seeds": list(config["permutation_seeds"]),
        "permuted_mae_ev": maes,
        "mean_mae_ev": mean_mae,
        "unpermuted_fold_0_mae_ev": unpermuted_mae,
        "passes": bool(mean_mae > unpermuted_mae),
    }


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("joblib", "matplotlib", "numpy", "scikit-learn"):
        versions[distribution] = importlib.metadata.version(distribution)
    return versions


def create_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            relative = str(path.relative_to(run_dir))
            files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload = {"schema_version": 1, "files": files}
    write_json(run_dir / "artifact_manifest.json", payload)
    return payload


def verify_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "artifact_manifest.json"
    expected = load_json(manifest_path)
    problems: list[str] = []
    for relative, metadata in expected["files"].items():
        path = run_dir / relative
        if not path.is_file():
            problems.append(f"missing: {relative}")
        elif sha256_file(path) != metadata["sha256"]:
            problems.append(f"hash mismatch: {relative}")
    observed = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    unexpected = observed - set(expected["files"])
    problems.extend(f"unmanifested: {relative}" for relative in sorted(unexpected))
    return {"ok": not problems, "problems": problems, "file_count": len(expected["files"])}


def run_study(root: Path, run_id: str) -> Path:
    # Import graphics only for a training/report run, never checkpoint inference.
    from .plotting import abstention_plot, calibration_plot, parity_plot, ranking_plot, subgroup_plot

    root = root.resolve()
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started_utc = _utc_now()
    start_time = time.monotonic()
    failures: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {"run_id": run_id, "started_utc": started_utc, "status": "RUNNING"}
    write_json(run_dir / "failures.json", failures)
    try:
        config = load_json(root / "configs/study.json")
        contract = load_data_contract(root, config["raw_data_contract"])
        guard = ResourceGuard(
            max_allowed_rss_mib=float(config["max_rss_mib"]),
            require_no_pageout_increase=bool(config["require_no_pageout_increase"]),
        )
        provenance_path = root / "data/manifests/provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(
                "prepared manifests are missing; run 'ml4materials prepare' before the study"
            )
        provenance = load_json(provenance_path)
        if (
            provenance.get("source_sha256") != contract["source_sha256"]
            or provenance.get("split_source_sha256") != contract["split_source_sha256"]
            or provenance.get("upstream_metadata_sha256")
            != contract["upstream_metadata_sha256"]
        ):
            raise ValueError("prepared provenance does not match the frozen data contract")
        records = load_records(root, contract)
        splits = load_official_splits(root, contract, records)
        guard.check("data_validation")

        formulas = [record.composition for record in records]
        targets = np.asarray([record.gap_ev for record in records], dtype=float)
        identifiers = [record.material_id for record in records]
        index_by_id = {identifier: index for index, identifier in enumerate(identifiers)}
        group_by_id = {
            record.material_id: record.canonical_composition for record in records
        }
        n_records = len(records)
        proposed = np.full(n_records, np.nan)
        ridge_predictions = np.full(n_records, np.nan)
        trivial_predictions = np.full(n_records, np.nan)
        raw_sigmas = np.full(n_records, np.nan)
        calibrated_sigmas = np.full(n_records, np.nan)
        fold_names = np.full(n_records, "", dtype=object)
        unseen_counts = np.zeros(n_records, dtype=int)
        seed_predictions = np.full(
            (n_records, len(config["model_seeds"])), np.nan, dtype=float
        )
        interval_bounds = {
            float(coverage): (
                np.full(n_records, np.nan),
                np.full(n_records, np.nan),
            )
            for coverage in config["interval_coverages"]
        }
        fold_metrics: list[dict[str, Any]] = []
        checkpoint_receipts: list[dict[str, Any]] = []

        for fold_name in config["official_folds"]:
            fold_started = time.monotonic()
            fold = splits[fold_name]
            train_ids = fold["train"]
            test_ids = fold["test"]
            train_indices = _indices(train_ids, index_by_id)
            test_indices = _indices(test_ids, index_by_id)
            fit_ids, calibration_ids = calibration_partition(
                train_ids,
                float(config["calibration_fraction"]),
                int(config["calibration_seed"]),
                group_by_id,
            )
            fit_indices = _indices(fit_ids, index_by_id)
            calibration_indices = _indices(calibration_ids, index_by_id)
            validate_group_isolation(fit_ids, calibration_ids, records)

            featurizer = CompositionFeaturizer().fit(
                [formulas[index] for index in fit_indices]
            )
            fit_features = featurizer.transform([formulas[index] for index in fit_indices])
            calibration_features = featurizer.transform(
                [formulas[index] for index in calibration_indices]
            )
            test_features = featurizer.transform([formulas[index] for index in test_indices])
            models = fit_ensemble(
                fit_features,
                targets[fit_indices],
                config["model_seeds"],
                config["extra_trees"],
            )
            calibration_prediction, calibration_raw_sigma, _ = ensemble_predictions(
                models, calibration_features
            )
            calibrator = NormalizedConformalCalibrator.fit(
                targets[calibration_indices],
                calibration_prediction,
                calibration_raw_sigma,
            )
            fold_prediction, fold_raw_sigma, fold_seed_predictions = ensemble_predictions(
                models, test_features
            )
            fold_calibrated_sigma = calibrator.calibrated_sigma(fold_raw_sigma)

            ridge_featurizer = CompositionFeaturizer().fit(
                [formulas[index] for index in train_indices]
            )
            ridge_train = ridge_featurizer.transform(
                [formulas[index] for index in train_indices]
            )
            ridge_test = ridge_featurizer.transform(
                [formulas[index] for index in test_indices]
            )
            ridge = make_ridge(config["ridge_alphas"])
            ridge.fit(ridge_train, targets[train_indices])
            fold_ridge = ridge.predict(ridge_test)
            fold_trivial = np.full(len(test_indices), np.median(targets[train_indices]))

            proposed[test_indices] = fold_prediction
            raw_sigmas[test_indices] = fold_raw_sigma
            calibrated_sigmas[test_indices] = fold_calibrated_sigma
            ridge_predictions[test_indices] = fold_ridge
            trivial_predictions[test_indices] = fold_trivial
            seed_predictions[test_indices, :] = fold_seed_predictions
            fold_names[test_indices] = fold_name
            unseen_counts[test_indices] = np.asarray(
                [
                    len(featurizer.unseen_elements(formulas[index]))
                    for index in test_indices
                ],
                dtype=int,
            )
            for coverage, (lower_all, upper_all) in interval_bounds.items():
                lower, upper, _ = calibrator.interval(
                    fold_prediction, fold_raw_sigma, coverage
                )
                lower_all[test_indices] = lower
                upper_all[test_indices] = upper

            checkpoint_path = run_dir / "models" / f"{fold_name}.joblib"
            save_checkpoint(
                checkpoint_path,
                {
                    "fold": fold_name,
                    "models": models,
                    "calibrator": calibrator.to_dict(),
                    "feature_names": list(featurizer.feature_names),
                    "train_element_counts": featurizer.train_element_counts,
                    "proper_train_ids": fit_ids,
                    "calibration_ids": calibration_ids,
                },
            )
            reloaded = load_checkpoint(checkpoint_path)
            reloaded_prediction, reloaded_sigma = _fold_checkpoint_prediction(
                reloaded, test_features
            )
            reload_ok = bool(
                np.array_equal(reloaded_prediction, fold_prediction)
                and np.array_equal(reloaded_sigma, fold_calibrated_sigma)
            )
            if not reload_ok:
                raise RuntimeError(f"checkpoint reload changed predictions for {fold_name}")
            checkpoint_receipts.append(
                {
                    "fold": fold_name,
                    "path": str(checkpoint_path.relative_to(run_dir)),
                    "sha256": sha256_file(checkpoint_path),
                    "reload_prediction_identity": reload_ok,
                }
            )
            fold_metrics.append(
                {
                    "fold": fold_name,
                    "train_n": int(len(train_indices)),
                    "proper_train_n": int(len(fit_indices)),
                    "calibration_n": int(len(calibration_indices)),
                    "test_n": int(len(test_indices)),
                    "proposed": regression_metrics(
                        targets[test_indices], fold_prediction
                    ),
                    "ridge": regression_metrics(targets[test_indices], fold_ridge),
                    "trivial": regression_metrics(targets[test_indices], fold_trivial),
                    "per_seed_mae_ev": [
                        float(
                            np.mean(
                                np.abs(
                                    targets[test_indices]
                                    - fold_seed_predictions[:, seed_column]
                                )
                            )
                        )
                        for seed_column in range(fold_seed_predictions.shape[1])
                    ],
                    "calibration_sigma_scale": calibrator.sigma_scale,
                    "calibration_sigma_floor": calibrator.sigma_floor,
                    "duration_seconds": time.monotonic() - fold_started,
                }
            )
            events.append(
                {
                    "event": "official_fold_complete",
                    "fold": fold_name,
                    "utc": _utc_now(),
                    "duration_seconds": time.monotonic() - fold_started,
                }
            )
            guard.check(f"official_{fold_name}")

        for array_name, array in {
            "proposed": proposed,
            "ridge": ridge_predictions,
            "trivial": trivial_predictions,
            "raw_sigma": raw_sigmas,
            "calibrated_sigma": calibrated_sigmas,
            "seed_predictions": seed_predictions,
        }.items():
            if not np.all(np.isfinite(array)):
                raise RuntimeError(f"non-finite values remain in out-of-fold {array_name}")

        official_metrics = {
            "proposed": regression_metrics(targets, proposed),
            "ridge": regression_metrics(targets, ridge_predictions),
            "trivial": regression_metrics(targets, trivial_predictions),
        }
        pooled_calibration = _pooled_calibration(targets, interval_bounds)
        abstention = abstention_metrics(
            targets,
            proposed,
            calibrated_sigmas,
            [float(value) for value in config["abstention_coverages"]],
        )
        subgroups = subgroup_metrics(
            targets, proposed, formulas, unseen_counts
        )

        pool_metadata, pool_targets, pool_predictions, pool_uncertainty = _candidate_pool(
            records, fold_names, targets, proposed, calibrated_sigmas
        )
        target_window = tuple(float(value) for value in config["candidate_target_window_ev"])
        ranking_metrics, proposed_order, target_probability = evaluate_ranking(
            pool_targets,
            pool_predictions,
            pool_uncertainty,
            target_window,  # type: ignore[arg-type]
            [int(value) for value in config["candidate_budgets"]],
            int(config["random_policy_replicates"]),
        )
        candidate_rows: list[dict[str, Any]] = []
        lower_target, upper_target = target_window
        for rank, pool_index in enumerate(proposed_order, start=1):
            metadata = pool_metadata[int(pool_index)]
            actual = float(pool_targets[pool_index])
            candidate_rows.append(
                {
                    "rank": rank,
                    **metadata,
                    "predicted_gap_ev": float(pool_predictions[pool_index]),
                    "calibrated_sigma_ev": float(pool_uncertainty[pool_index]),
                    "target_window_probability": float(target_probability[pool_index]),
                    "evaluation_mean_gap_ev": actual,
                    "evaluation_hit_1p1_to_1p6_ev": bool(
                        lower_target <= actual <= upper_target
                    ),
                    "claim_boundary": "ranked historical prediction; not stability or utility verification",
                }
            )
        guard.check("candidate_ranking")

        generalization = _generalization_study(records, config, group_by_id, guard)
        fold_0_mae = next(
            row["proposed"]["mae_ev"] for row in fold_metrics if row["fold"] == "fold_0"
        )
        permutation = _permutation_control(
            records, splits, config, group_by_id, float(fold_0_mae)
        )
        guard.check("label_permutation")

        results = {
            "schema_version": 1,
            "status": "EXECUTED_DEVELOPMENT_NOT_INDEPENDENTLY_VALIDATED",
            "study_version": config["study_version"],
            "objective": config["objective"],
            "dataset": {
                **provenance,
                "duplicate_groups": len(duplicate_audit(records)),
                "official_split_leakage": split_leakage_audit(splits, records),
            },
            "official_cv": {
                "metrics": official_metrics,
                "fold_metrics": fold_metrics,
                "calibration": pooled_calibration,
                "abstention": abstention,
                "subgroups": subgroups,
            },
            "candidate_ranking": {
                "pool_size": len(pool_targets),
                "target_window_ev": list(target_window),
                "pool_prevalence": float(
                    np.mean((pool_targets >= lower_target) & (pool_targets <= upper_target))
                ),
                "metrics": ranking_metrics,
            },
            "generalization": generalization,
            "sanity_checks": {
                "label_permutation": permutation,
                "checkpoint_reload_all_passed": all(
                    row["reload_prediction_identity"] for row in checkpoint_receipts
                ),
                "checkpoint_receipts": checkpoint_receipts,
            },
        }

        prediction_rows: list[dict[str, Any]] = []
        lower_90, upper_90 = interval_bounds[0.9]
        for index, record in enumerate(records):
            prediction_rows.append(
                {
                    "material_id": record.material_id,
                    "composition": record.composition,
                    "canonical_composition": record.canonical_composition,
                    "fold": str(fold_names[index]),
                    "target_gap_ev": float(targets[index]),
                    "proposed_prediction_ev": float(proposed[index]),
                    "ridge_prediction_ev": float(ridge_predictions[index]),
                    "trivial_prediction_ev": float(trivial_predictions[index]),
                    "raw_tree_sigma_ev": float(raw_sigmas[index]),
                    "calibrated_sigma_ev": float(calibrated_sigmas[index]),
                    "interval_90_lower_ev": float(lower_90[index]),
                    "interval_90_upper_ev": float(upper_90[index]),
                    "unseen_element_count_vs_proper_train": int(unseen_counts[index]),
                    **{
                        f"seed_{seed}_prediction_ev": float(
                            seed_predictions[index, seed_column]
                        )
                        for seed_column, seed in enumerate(config["model_seeds"])
                    },
                }
            )

        write_json(run_dir / "results.json", results)
        write_json(run_dir / "events.json", events)
        write_json(run_dir / "failures.json", failures)
        _write_csv(run_dir / "tables/official_fold_metrics.csv", fold_metrics)
        _write_csv(run_dir / "tables/calibration.csv", pooled_calibration)
        _write_csv(run_dir / "tables/abstention.csv", abstention)
        _write_csv(run_dir / "tables/subgroups.csv", subgroups)
        _write_csv(run_dir / "tables/ranking.csv", ranking_metrics)
        _write_csv(run_dir / "tables/generalization.csv", generalization)
        _write_csv(run_dir / "predictions/oof_predictions.csv", prediction_rows)
        _write_csv(run_dir / "predictions/candidate_ledger.csv", candidate_rows)

        figures = run_dir / "figures"
        parity_plot(targets, proposed, ridge_predictions, figures / "parity.png")
        calibration_plot(pooled_calibration, figures / "calibration.png")
        abstention_plot(abstention, figures / "abstention.png")
        ranking_plot(ranking_metrics, figures / "ranking.png")
        subgroup_plot(subgroups, figures / "subgroups.png")
        guard.check("figure_generation")

        finished_utc = _utc_now()
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "duration_seconds": time.monotonic() - start_time,
            "command": f"ml4materials run --run-id {run_id}",
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "dependencies": _installed_versions(),
            "inputs": {
                "dataset_sha256": contract["source_sha256"],
                "split_sha256": contract["split_source_sha256"],
                "metadata_sha256": contract["upstream_metadata_sha256"],
                "study_config_sha256": sha256_file(root / "configs/study.json"),
                "data_contract_sha256": sha256_file(root / config["raw_data_contract"]),
            },
            "git": {
                "root": _git_value(root, ["rev-parse", "--show-toplevel"]),
                "head": _git_value(root, ["rev-parse", "HEAD"]),
                "scoped_status": _git_value(root, ["status", "--short", "--", "."]),
            },
            "resources": guard.receipt(),
        }
        write_json(run_dir / "receipt.json", receipt)
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        results_markdown = render_results(results, receipt)
        manuscript_markdown = render_manuscript(results, receipt)
        (reports_dir / "RESULTS.md").write_text(results_markdown, encoding="utf-8")
        (reports_dir / "MANUSCRIPT.md").write_text(manuscript_markdown, encoding="utf-8")
        (root / "RESULTS.md").write_text(results_markdown, encoding="utf-8")
        (root / "MANUSCRIPT.md").write_text(manuscript_markdown, encoding="utf-8")
        create_artifact_manifest(run_dir)
        return run_dir
    except Exception as error:
        failures.append(
            {
                "utc": _utc_now(),
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(run_dir / "failures.json", failures)
        receipt.update(
            {
                "status": "FAILED_RETAINED",
                "finished_utc": _utc_now(),
                "duration_seconds": time.monotonic() - start_time,
                "completed_events": events,
                "completed_checkpoint_files": [
                    str(path.relative_to(run_dir))
                    for path in sorted((run_dir / "models").glob("*.joblib"))
                ]
                if (run_dir / "models").is_dir()
                else [],
            }
        )
        if "guard" in locals():
            receipt["resources"] = guard.receipt()
        if "contract" in locals():
            receipt["inputs"] = {
                "dataset_sha256": contract.get("source_sha256"),
                "split_sha256": contract.get("split_source_sha256"),
                "metadata_sha256": contract.get("upstream_metadata_sha256"),
            }
        write_json(run_dir / "events.json", events)
        write_json(run_dir / "receipt.json", receipt)
        create_artifact_manifest(run_dir)
        raise
