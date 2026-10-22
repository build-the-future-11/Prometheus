"""Checkpoint-backed prediction, calibrated uncertainty, and label-blind ranking."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .features import CompositionFeaturizer
from .models import ensemble_predictions, load_checkpoint
from .uncertainty import NormalizedConformalCalibrator, normal_target_probability


def predict_formulas(
    checkpoint_path: Path,
    formulas: list[str],
    target_window: tuple[float, float] = (1.1, 1.6),
    interval_coverage: float = 0.9,
) -> list[dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    featurizer = CompositionFeaturizer(
        train_element_counts={
            str(key): int(value)
            for key, value in checkpoint["train_element_counts"].items()
        },
        fitted=True,
    )
    if list(featurizer.feature_names) != checkpoint["feature_names"]:
        raise ValueError("checkpoint feature schema does not match this package version")
    features = featurizer.transform(formulas)
    prediction, raw_sigma, _ = ensemble_predictions(checkpoint["models"], features)
    calibrator = NormalizedConformalCalibrator.from_dict(checkpoint["calibrator"])
    calibrated_sigma = calibrator.calibrated_sigma(raw_sigma)
    lower, upper, _ = calibrator.interval(prediction, raw_sigma, interval_coverage)
    probability = normal_target_probability(
        prediction, calibrated_sigma, target_window[0], target_window[1]
    )
    return [
        {
            "composition": formula,
            "predicted_gap_ev": float(prediction[index]),
            "raw_tree_sigma_ev": float(raw_sigma[index]),
            "calibrated_sigma_ev": float(calibrated_sigma[index]),
            f"interval_{int(interval_coverage * 100)}_lower_ev": float(lower[index]),
            f"interval_{int(interval_coverage * 100)}_upper_ev": float(upper[index]),
            "target_window_probability": float(probability[index]),
            "unseen_elements": list(featurizer.unseen_elements(formula)),
        }
        for index, formula in enumerate(formulas)
    ]


def rank_csv(
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    target_window: tuple[float, float] = (1.1, 1.6),
) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "composition" not in rows[0]:
        raise ValueError("candidate CSV must contain a non-empty 'composition' column")
    formulas = [row["composition"] for row in rows]
    predictions = predict_formulas(checkpoint_path, formulas, target_window)
    combined = [{**source, **prediction} for source, prediction in zip(rows, predictions)]
    combined.sort(key=lambda row: float(row["target_window_probability"]), reverse=True)
    for rank, row in enumerate(combined, start=1):
        row["rank"] = rank
    fieldnames = ["rank", *[key for key in combined[0] if key != "rank"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    temporary.replace(output_path)


def inspect_checkpoint(checkpoint_path: Path, limit: int = 20) -> dict[str, Any]:
    """Summarize model-attributed feature importance without accessing labels."""

    if limit < 1:
        raise ValueError("limit must be at least one")
    checkpoint = load_checkpoint(checkpoint_path)
    feature_names = [str(name) for name in checkpoint["feature_names"]]
    importances = np.asarray(
        [model.feature_importances_ for model in checkpoint["models"]], dtype=float
    )
    if importances.ndim != 2 or importances.shape[1] != len(feature_names):
        raise ValueError("checkpoint model importances do not match feature schema")
    order = np.argsort(-importances.mean(axis=0), kind="mergesort")[:limit]
    return {
        "checkpoint_type": "extra_trees_conformal",
        "fold": checkpoint.get("fold"),
        "model_count": len(checkpoint["models"]),
        "feature_count": len(feature_names),
        "top_features": [
            {
                "feature": feature_names[int(index)],
                "mean_importance": float(importances[:, index].mean()),
                "std_importance": float(importances[:, index].std(ddof=0)),
            }
            for index in order
        ],
        "claim_boundary": "importance describes fitted model behavior, not causal chemistry",
    }

