"""SPARKING model and compute-matched controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RIDGE_ALPHAS = np.logspace(-4, 4, 17)


def make_ridge() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )


def make_hgb(config: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=float(config["learning_rate"]),
        max_iter=int(config["max_iter"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=float(config["l2_regularization"]),
        random_state=int(config["random_state"]),
        early_stopping=False,
    )


@dataclass
class SparkingRegressor:
    """Linear prediction plus support-gated nonlinear residual correction."""

    ridge: Pipeline
    residual_model: HistGradientBoostingRegressor
    support_scaler: StandardScaler
    neighbours: NearestNeighbors
    leave_one_out_distances: np.ndarray

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        config: dict[str, Any],
    ) -> "SparkingRegressor":
        if features.ndim != 2 or targets.shape != (features.shape[0],):
            raise ValueError("invalid feature/target shapes")
        ridge = make_ridge()
        ridge.fit(features, targets)
        residuals = targets - np.asarray(ridge.predict(features), dtype=float)
        residual_model = make_hgb(config)
        residual_model.fit(features, residuals)
        support_scaler = StandardScaler()
        support_features = support_scaler.fit_transform(features)
        neighbours = NearestNeighbors(n_neighbors=2, metric="euclidean", algorithm="auto")
        neighbours.fit(support_features)
        distances, _ = neighbours.kneighbors(support_features, n_neighbors=2)
        leave_one_out = np.asarray(distances[:, 1], dtype=float)
        if not np.all(np.isfinite(leave_one_out)):
            raise ValueError("support distances are non-finite")
        return cls(ridge, residual_model, support_scaler, neighbours, leave_one_out)

    def components(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        baseline = np.asarray(self.ridge.predict(features), dtype=float)
        residual = np.asarray(self.residual_model.predict(features), dtype=float)
        scaled = self.support_scaler.transform(features)
        distance = np.asarray(self.neighbours.kneighbors(scaled, n_neighbors=1)[0][:, 0], dtype=float)
        if not all(np.all(np.isfinite(values)) for values in (baseline, residual, distance)):
            raise ValueError("model produced non-finite output")
        return baseline, residual, distance

    def threshold(self, support_quantile: float) -> float:
        if not 0.0 < support_quantile < 1.0:
            raise ValueError("support_quantile must be between zero and one")
        value = float(np.quantile(self.leave_one_out_distances, support_quantile))
        if not np.isfinite(value) or value <= 0:
            raise ValueError("invalid support threshold")
        return value

    def predict_ungated(self, features: np.ndarray) -> np.ndarray:
        baseline, residual, _ = self.components(features)
        return np.clip(baseline + residual, 0.0, None)

    def predict(
        self,
        features: np.ndarray,
        support_quantile: float,
        power: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if power <= 0:
            raise ValueError("gate power must be positive")
        baseline, residual, distance = self.components(features)
        threshold = self.threshold(support_quantile)
        denominator = np.maximum(distance, threshold)
        gate = np.clip((threshold / denominator) ** power, 0.0, 1.0)
        prediction = np.clip(baseline + gate * residual, 0.0, None)
        return prediction, gate, distance


def direct_hgb_predictions(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    model = make_hgb(config)
    model.fit(train_features, train_targets)
    return np.clip(np.asarray(model.predict(test_features), dtype=float), 0.0, None)

