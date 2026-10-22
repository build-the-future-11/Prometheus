"""Prespecified prediction, calibration, subgroup, and abstention endpoints."""

from __future__ import annotations

from math import sqrt
from typing import Callable

import numpy as np

from .chemistry import element_fractions
from .uncertainty import NormalizedConformalCalibrator


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("metric arrays must be non-empty and aligned")
    residual = targets - predictions
    mae = float(np.mean(np.abs(residual)))
    rmse = float(sqrt(float(np.mean(residual**2))))
    denominator = float(np.sum((targets - targets.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator else 0.0
    target_ranks = _rankdata(targets)
    prediction_ranks = _rankdata(predictions)
    if np.std(target_ranks) == 0 or np.std(prediction_ranks) == 0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(target_ranks, prediction_ranks)[0, 1])
    return {"mae_ev": mae, "rmse_ev": rmse, "r2": r2, "spearman": spearman}


def calibration_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    raw_sigma: np.ndarray,
    calibrator: NormalizedConformalCalibrator,
    coverages: list[float],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for coverage in coverages:
        lower, upper, quantile = calibrator.interval(predictions, raw_sigma, coverage)
        observed = float(np.mean((targets >= lower) & (targets <= upper)))
        rows.append(
            {
                "nominal_coverage": float(coverage),
                "observed_coverage": observed,
                "coverage_error": observed - float(coverage),
                "mean_interval_width_ev": float(np.mean(upper - lower)),
                "normalized_score_quantile": quantile,
            }
        )
    return rows


def interval_metrics(
    targets: np.ndarray, lower: np.ndarray, upper: np.ndarray, coverage: float
) -> dict[str, float]:
    """Evaluate arbitrary prediction intervals with a proper interval score."""

    if not (0.0 < coverage < 1.0):
        raise ValueError("coverage must lie strictly between zero and one")
    if not (len(targets) == len(lower) == len(upper)) or not len(targets):
        raise ValueError("interval arrays must be non-empty and aligned")
    ordered_lower = np.minimum(lower, upper)
    ordered_upper = np.maximum(lower, upper)
    alpha = 1.0 - coverage
    width = ordered_upper - ordered_lower
    score = width + (2.0 / alpha) * np.maximum(ordered_lower - targets, 0.0)
    score += (2.0 / alpha) * np.maximum(targets - ordered_upper, 0.0)
    return {
        "nominal_coverage": float(coverage),
        "observed_coverage": float(
            np.mean((targets >= ordered_lower) & (targets <= ordered_upper))
        ),
        "mean_interval_width_ev": float(np.mean(width)),
        "mean_interval_score_ev": float(np.mean(score)),
    }


def abstention_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    retained_fractions: list[float],
) -> list[dict[str, float]]:
    order = np.argsort(uncertainty, kind="mergesort")
    overall_mae = float(np.mean(np.abs(targets - predictions)))
    rows: list[dict[str, float]] = []
    for retained_fraction in retained_fractions:
        count = max(1, int(round(len(targets) * retained_fraction)))
        retained = order[:count]
        rejected = order[count:]
        retained_mae = float(np.mean(np.abs(targets[retained] - predictions[retained])))
        rejected_mae = (
            float(np.mean(np.abs(targets[rejected] - predictions[rejected])))
            if len(rejected)
            else None
        )
        rows.append(
            {
                "retained_fraction": float(retained_fraction),
                "retained_count": int(count),
                "retained_mae_ev": retained_mae,
                "rejected_mae_ev": rejected_mae,
                "mae_change_vs_no_abstention_ev": retained_mae - overall_mae,
            }
        )
    return rows


def subgroup_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    formulas: list[str],
    unseen_counts: np.ndarray,
) -> list[dict[str, float | int | str]]:
    parsed = [element_fractions(formula) for formula in formulas]
    selectors: dict[str, Callable[[int], bool]] = {
        "binary": lambda i: len(parsed[i]) == 2,
        "ternary": lambda i: len(parsed[i]) == 3,
        "quaternary_or_more": lambda i: len(parsed[i]) >= 4,
        "contains_oxygen": lambda i: "O" in parsed[i],
        "no_oxygen": lambda i: "O" not in parsed[i],
        "all_elements_seen_in_proper_train": lambda i: unseen_counts[i] == 0,
        "has_unseen_element": lambda i: unseen_counts[i] > 0,
    }
    rows: list[dict[str, float | int | str]] = []
    for name, selector in selectors.items():
        indices = np.asarray([index for index in range(len(formulas)) if selector(index)])
        if not len(indices):
            continue
        row: dict[str, float | int | str] = {"subgroup": name, "n": int(len(indices))}
        row.update(regression_metrics(targets[indices], predictions[indices]))
        rows.append(row)
    return rows

