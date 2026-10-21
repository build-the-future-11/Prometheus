"""Deterministic regression and interval metrics."""

from __future__ import annotations

import numpy as np


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("targets and predictions must be equal one-dimensional arrays")
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(prediction))):
        raise ValueError("metrics require finite inputs")
    error = prediction - target
    return {
        "n": int(target.size),
        "mae_ev": float(np.mean(np.abs(error))),
        "rmse_ev": float(np.sqrt(np.mean(error**2))),
        "median_ae_ev": float(np.median(np.abs(error))),
        "bias_ev": float(np.mean(error)),
    }


def conformal_radius(calibration_errors: np.ndarray, coverage: float) -> float:
    errors = np.sort(np.asarray(calibration_errors, dtype=float))
    if errors.ndim != 1 or errors.size < 2 or not np.all(np.isfinite(errors)):
        raise ValueError("invalid calibration errors")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie between zero and one")
    rank = int(np.ceil((errors.size + 1) * coverage))
    rank = min(max(rank, 1), errors.size)
    return float(errors[rank - 1])


def interval_metrics(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: float,
) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if target.shape != lower.shape or target.shape != upper.shape:
        raise ValueError("interval arrays must have equal shapes")
    if np.any(lower > upper):
        raise ValueError("lower interval exceeds upper interval")
    alpha = 1.0 - coverage
    width = upper - lower
    score = width.copy()
    score += (2.0 / alpha) * (lower - target) * (target < lower)
    score += (2.0 / alpha) * (target - upper) * (target > upper)
    return {
        "nominal_coverage": float(coverage),
        "observed_coverage": float(np.mean((target >= lower) & (target <= upper))),
        "mean_width_ev": float(np.mean(width)),
        "mean_interval_score_ev": float(np.mean(score)),
    }


def bootstrap_mean_interval(values: np.ndarray, seed: int = 20261020, resamples: int = 10000) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap requires at least two finite values")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "resamples": int(resamples),
        "seed": int(seed),
    }

