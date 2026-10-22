"""Model construction, ensemble uncertainty, and versioned checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CHECKPOINT_SCHEMA_VERSION = 1
QUANTILE_CHECKPOINT_SCHEMA_VERSION = 1


def make_ridge(alphas: Iterable[float]) -> Pipeline:
    """Regularized linear composition baseline with train-fitted scaling."""

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.asarray(tuple(alphas), dtype=float))),
        ]
    )


def make_extra_trees(seed: int, config: dict[str, Any]) -> ExtraTreesRegressor:
    """Construct the frozen nonlinear composition model."""

    return ExtraTreesRegressor(
        n_estimators=int(config["n_estimators_per_seed"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=float(config["max_features"]),
        n_jobs=int(config["n_jobs"]),
        random_state=int(seed),
        bootstrap=False,
    )


def fit_ensemble(
    features: np.ndarray,
    targets: np.ndarray,
    seeds: Iterable[int],
    config: dict[str, Any],
) -> list[ExtraTreesRegressor]:
    models: list[ExtraTreesRegressor] = []
    for seed in seeds:
        model = make_extra_trees(int(seed), config)
        model.fit(features, targets)
        models.append(model)
    return models


def make_quantile_gradient_boosting(
    quantile: float, config: dict[str, Any]
) -> GradientBoostingRegressor:
    """Create a deterministic quantile baseline for successor studies.

    This model is deliberately separate from the frozen ExtraTrees protocol.
    It estimates conditional quantiles directly; any future coverage claim must
    still be evaluated on untouched outer-test rows.
    """

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=float(quantile),
        n_estimators=int(config.get("n_estimators", 160)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        max_depth=int(config.get("max_depth", 3)),
        min_samples_leaf=int(config.get("min_samples_leaf", 4)),
        subsample=float(config.get("subsample", 1.0)),
        random_state=int(config.get("seed", 20261020)),
    )


def fit_quantile_interval_models(
    features: np.ndarray,
    targets: np.ndarray,
    coverage: float,
    config: dict[str, Any],
) -> tuple[GradientBoostingRegressor, GradientBoostingRegressor]:
    """Fit lower/upper quantile models for a requested central interval."""

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between zero and one")
    tail = (1.0 - coverage) / 2.0
    lower = make_quantile_gradient_boosting(tail, config)
    upper = make_quantile_gradient_boosting(1.0 - tail, config)
    lower.fit(features, targets)
    upper.fit(features, targets)
    return lower, upper


def quantile_interval_predictions(
    lower_model: GradientBoostingRegressor,
    upper_model: GradientBoostingRegressor,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered quantile bounds and reject non-finite model output."""

    lower = np.asarray(lower_model.predict(features), dtype=float)
    upper = np.asarray(upper_model.predict(features), dtype=float)
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("quantile model produced non-finite interval bounds")
    return np.minimum(lower, upper), np.maximum(lower, upper)


def ensemble_predictions(
    models: list[ExtraTreesRegressor], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, tree-level standard deviation, and per-seed predictions."""

    if not models:
        raise ValueError("at least one fitted model is required")
    per_seed = np.column_stack([model.predict(features) for model in models])
    tree_predictions = np.column_stack(
        [
            estimator.predict(features).astype(np.float32, copy=False)
            for model in models
            for estimator in model.estimators_
        ]
    )
    mean = tree_predictions.mean(axis=1, dtype=np.float64)
    raw_sigma = tree_predictions.std(axis=1, ddof=1, dtype=np.float64)
    return mean, raw_sigma, per_seed


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = {"schema_version": CHECKPOINT_SCHEMA_VERSION, **payload}
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(complete, temporary, compress=3)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema drift: {payload.get('schema_version')!r} != {CHECKPOINT_SCHEMA_VERSION}"
        )
    required = {"models", "calibrator", "feature_names", "train_element_counts"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    return payload


def save_quantile_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Persist a successor quantile-model checkpoint atomically."""

    required = {"point_model", "interval_models", "feature_names", "train_element_counts"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"quantile checkpoint missing fields: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = {"schema_version": QUANTILE_CHECKPOINT_SCHEMA_VERSION, **payload}
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(complete, temporary, compress=3)
    temporary.replace(path)


def load_quantile_checkpoint(path: Path) -> dict[str, Any]:
    """Load a versioned quantile-model checkpoint and reject schema drift."""

    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("quantile checkpoint payload must be a dictionary")
    if payload.get("schema_version") != QUANTILE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("quantile checkpoint schema drift")
    required = {"point_model", "interval_models", "feature_names", "train_element_counts"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"quantile checkpoint missing fields: {sorted(missing)}")
    return payload

