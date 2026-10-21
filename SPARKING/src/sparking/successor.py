"""Versioned SPARKING successor models and training-only risk calibration.

This module is additive.  It does not alter the frozen v1 model or protocol.
The v2 mechanism combines a strong direct composition model with ridge through
a gate trained only on pseudo-element exclusions inside the outer training set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from catboost import CatBoostRegressor
from matminer.featurizers.composition import ElementProperty, Stoichiometry
import numpy as np
from pymatgen.core import Composition
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .chemistry import ELEMENTS, element_fractions, featurize
from .model import RIDGE_ALPHAS


@dataclass(frozen=True)
class FeatureMatrix:
    values: np.ndarray
    base_values: np.ndarray
    names: tuple[str, ...]


def publication_features(formulas: Iterable[str]) -> FeatureMatrix:
    """Build fixed Magpie, stoichiometric, and v1 composition descriptors."""

    formulas = tuple(formulas)
    magpie = ElementProperty.from_preset("magpie")
    stoichiometry = Stoichiometry()
    rows: list[list[float]] = []
    for formula in formulas:
        composition = Composition(formula)
        rows.append(
            [float(value) for value in magpie.featurize(composition)]
            + [float(value) for value in stoichiometry.featurize(composition)]
        )
    mined = np.asarray(rows, dtype=float)
    base = featurize(formulas)
    values = np.hstack([base, mined])
    if values.ndim != 2 or values.shape[0] != len(formulas):
        raise ValueError("invalid publication feature shape")
    if np.any(np.isinf(values)):
        raise ValueError("publication features contain infinities")
    names = (
        tuple(f"v1_{index}" for index in range(base.shape[1]))
        + tuple(f"magpie_{label}" for label in magpie.feature_labels())
        + tuple(f"stoichiometry_{label}" for label in stoichiometry.feature_labels())
    )
    if len(names) != values.shape[1]:
        raise ValueError("publication feature labels do not match matrix")
    return FeatureMatrix(values=values, base_values=base, names=names)


def make_publication_ridge() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )


def _catboost_params(config: dict[str, Any], seed: int, task_type: str) -> dict[str, Any]:
    task = task_type.upper()
    if task not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU")
    params: dict[str, Any] = {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "iterations": int(config["iterations"]),
        "depth": int(config["depth"]),
        "learning_rate": float(config["learning_rate"]),
        "l2_leaf_reg": float(config["l2_leaf_reg"]),
        "random_strength": float(config["random_strength"]),
        "random_seed": int(seed),
        "task_type": task,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": int(config.get("thread_count", -1)),
    }
    if task == "GPU":
        params.update(
            {
                "devices": str(config.get("devices", "0")),
                "border_count": int(config.get("border_count", 128)),
                "bootstrap_type": "Bayesian",
                "bagging_temperature": float(config.get("bagging_temperature", 1.0)),
            }
        )
    return params


@dataclass(frozen=True)
class EnsemblePrediction:
    mean: np.ndarray
    standard_deviation: np.ndarray
    members: np.ndarray


def catboost_ensemble_predictions(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    config: dict[str, Any],
    seeds: Iterable[int],
    task_type: str,
) -> EnsemblePrediction:
    train_imputer = SimpleImputer(strategy="median")
    train = train_imputer.fit_transform(train_features)
    test = train_imputer.transform(test_features)
    member_predictions = []
    for seed in seeds:
        model = CatBoostRegressor(**_catboost_params(config, int(seed), task_type))
        model.fit(train, train_targets)
        member_predictions.append(np.clip(np.asarray(model.predict(test), dtype=float), 0.0, None))
    members = np.asarray(member_predictions, dtype=float)
    if members.ndim != 2 or members.shape[1] != test_features.shape[0]:
        raise ValueError("invalid CatBoost ensemble predictions")
    if not np.all(np.isfinite(members)):
        raise ValueError("CatBoost produced non-finite predictions")
    return EnsemblePrediction(
        mean=np.mean(members, axis=0),
        standard_deviation=np.std(members, axis=0),
        members=members,
    )


def extra_trees_ensemble_predictions(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    config: dict[str, Any],
    seeds: Iterable[int],
) -> EnsemblePrediction:
    imputer = SimpleImputer(strategy="median")
    train = imputer.fit_transform(train_features)
    test = imputer.transform(test_features)
    members = []
    for seed in seeds:
        model = ExtraTreesRegressor(
            n_estimators=int(config["n_estimators"]),
            max_features=float(config["max_features"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            n_jobs=-1,
            random_state=int(seed),
        )
        model.fit(train, train_targets)
        members.append(np.clip(np.asarray(model.predict(test), dtype=float), 0.0, None))
    matrix = np.asarray(members, dtype=float)
    return EnsemblePrediction(np.mean(matrix, axis=0), np.std(matrix, axis=0), matrix)


@dataclass
class RiskContext:
    scaler: StandardScaler
    neighbours: NearestNeighbors
    imputer: SimpleImputer
    observed_elements: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, base_features: np.ndarray) -> "RiskContext":
        imputer = SimpleImputer(strategy="median")
        clean = imputer.fit_transform(features)
        scaler = StandardScaler().fit(clean)
        scaled = scaler.transform(clean)
        neighbours = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(scaled)
        observed = np.any(base_features[:, : len(ELEMENTS)] > 0.0, axis=0)
        return cls(scaler, neighbours, imputer, observed)

    def transform(
        self,
        features: np.ndarray,
        base_features: np.ndarray,
        direct_mean: np.ndarray,
        direct_std: np.ndarray,
        ridge_prediction: np.ndarray,
    ) -> np.ndarray:
        clean = self.imputer.transform(features)
        scaled = self.scaler.transform(clean)
        distance = self.neighbours.kneighbors(scaled, n_neighbors=1)[0][:, 0]
        fractions = base_features[:, : len(ELEMENTS)]
        unseen_mask = (~self.observed_elements)[None, :] & (fractions > 0.0)
        unseen_fraction = np.sum(np.where(unseen_mask, fractions, 0.0), axis=1)
        unseen_count = np.sum(unseen_mask, axis=1).astype(float)
        n_elements = base_features[:, len(ELEMENTS)]
        entropy = base_features[:, len(ELEMENTS) + 1]
        matrix = np.column_stack(
            [
                direct_mean,
                direct_std,
                ridge_prediction,
                np.abs(direct_mean - ridge_prediction),
                distance,
                unseen_fraction,
                unseen_count,
                n_elements,
                entropy,
            ]
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("risk features are non-finite")
        return matrix


RISK_FEATURE_NAMES = (
    "direct_mean_ev",
    "ensemble_std_ev",
    "ridge_ev",
    "direct_ridge_disagreement_ev",
    "nearest_training_distance",
    "unseen_element_fraction",
    "unseen_element_count",
    "element_count",
    "fraction_entropy",
)


def fit_harm_model(
    risk_features: np.ndarray,
    direct_harm: np.ndarray,
    sample_weights: np.ndarray,
    config: dict[str, Any],
) -> HistGradientBoostingRegressor:
    if risk_features.shape[0] != direct_harm.size or direct_harm.size != sample_weights.size:
        raise ValueError("harm calibration shapes do not match")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(config["learning_rate"]),
        max_iter=int(config["max_iter"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=float(config["l2_regularization"]),
        random_state=int(config["random_state"]),
        early_stopping=False,
    )
    model.fit(risk_features, direct_harm, sample_weight=sample_weights)
    return model


def risk_gate_weights(predicted_harm: np.ndarray, temperature: float) -> np.ndarray:
    """Return direct-model weights; only predicted positive harm triggers shrinkage."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    harm = np.asarray(predicted_harm, dtype=float)
    if not np.all(np.isfinite(harm)):
        raise ValueError("predicted harm is non-finite")
    return np.exp(-np.maximum(harm, 0.0) / float(temperature))


def convex_prediction(ridge: np.ndarray, direct: np.ndarray, direct_weight: np.ndarray | float) -> np.ndarray:
    weight = np.asarray(direct_weight, dtype=float)
    if np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError("direct weights must lie in [0, 1]")
    return np.clip(ridge + weight * (direct - ridge), 0.0, None)


def weighted_spearman(values: np.ndarray, targets: np.ndarray) -> float | None:
    if np.std(values) == 0.0 or np.std(targets) == 0.0:
        return None
    result = float(spearmanr(values, targets).statistic)
    return result if np.isfinite(result) else None


def formula_element_mask(formulas: Iterable[str], element: str) -> np.ndarray:
    return np.asarray([element in element_fractions(formula) for formula in formulas], dtype=bool)
