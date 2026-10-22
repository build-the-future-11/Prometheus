"""Causal online forecasters used by the registered study.

The implementation intentionally keeps the proposed and control ensembles in one
class so they share experts, features, predictions, and causal losses. The ``mode``
switch selects the aggregation update. Revision-2 controls separately disable the
adaptive share rate and directed redistribution target, making each mechanism
contrast explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Literal, Sequence

import numpy as np


Mode = Literal[
    "uniform",
    "hedge",
    "adahedge",
    "fixed_share",
    "adaptive_share",
    "directed_fixed_share",
    "ddfs",
]


@dataclass(frozen=True)
class DDFSConfig:
    forgetting_factors: tuple[float, ...] = (0.90, 0.97, 0.995)
    lags: tuple[int, ...] = (1, 2, 3)
    ridge: float = 10.0
    eta: float = 0.35
    base_share: float = 0.01
    max_share: float = 0.03
    drift_gain: float = 0.02
    fast_rate: float = 0.20
    slow_rate: float = 0.02
    prediction_clip: float = 8.0
    loss_clip: float = 8.0
    min_scale: float = 1e-6
    interval_alpha: float = 0.10
    interval_window: int = 256

    def validate(self) -> None:
        if not self.forgetting_factors:
            raise ValueError("at least one forgetting factor is required")
        if any(not 0.0 < value <= 1.0 for value in self.forgetting_factors):
            raise ValueError("forgetting factors must be in (0, 1]")
        if any(lag <= 0 for lag in self.lags):
            raise ValueError("lags must be positive")
        if tuple(sorted(set(self.lags))) != self.lags:
            raise ValueError("lags must be strictly increasing")
        if self.ridge <= 0 or self.eta <= 0:
            raise ValueError("ridge and eta must be positive")
        if not 0 <= self.base_share <= self.max_share < 1:
            raise ValueError("shares must satisfy 0 <= base <= max < 1")
        if not 0 < self.fast_rate <= 1 or not 0 < self.slow_rate <= 1:
            raise ValueError("EWMA rates must be in (0, 1]")
        if self.fast_rate <= self.slow_rate:
            raise ValueError("fast_rate must exceed slow_rate")
        if not 0 < self.interval_alpha < 1:
            raise ValueError("interval_alpha must be in (0, 1)")


class RecursiveLeastSquares:
    """Forgetting-factor RLS with finite-state safeguards."""

    def __init__(self, dimension: int, forgetting: float, ridge: float) -> None:
        self.dimension = dimension
        self.forgetting = forgetting
        self.ridge = ridge
        self.theta = np.zeros(dimension, dtype=np.float64)
        self.covariance = np.eye(dimension, dtype=np.float64) / ridge
        self.reset_count = 0

    def predict(self, features: np.ndarray) -> float:
        value = float(self.theta @ features)
        return value if isfinite(value) else 0.0

    def update(self, features: np.ndarray, target: float) -> None:
        covariance_x = self.covariance @ features
        denominator = self.forgetting + float(features @ covariance_x)
        if denominator <= 1e-12 or not isfinite(denominator):
            self._reset()
            return
        gain = covariance_x / denominator
        error = target - float(self.theta @ features)
        theta = self.theta + gain * error
        covariance = (self.covariance - np.outer(gain, covariance_x)) / self.forgetting
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(covariance)):
            self._reset()
            return
        diagonal = np.diag(covariance)
        if np.any(diagonal <= 0) or float(np.max(diagonal)) > 1e12:
            self._reset()
            return
        self.theta = theta
        self.covariance = covariance

    def _reset(self) -> None:
        self.theta.fill(0.0)
        self.covariance = np.eye(self.dimension, dtype=np.float64) / self.ridge
        self.reset_count += 1


@dataclass(frozen=True)
class Prediction:
    value: float
    lower: float
    upper: float
    base: float
    scale: float
    expert_values: tuple[float, ...]
    weights: tuple[float, ...]
    drift_score: float
    share: float


class DriftDirectedFixedShare:
    """Multi-timescale RLS ensemble with matched mechanism-off modes.

    ``predict`` never reads the current outcome.  ``update`` consumes the outcome only
    after the prediction object has been created, so the API enforces predict-then-
    update ordering.
    """

    def __init__(
        self,
        *,
        season: int,
        config: DDFSConfig | None = None,
        mode: Mode = "ddfs",
    ) -> None:
        if season <= 0:
            raise ValueError("season must be positive")
        if mode not in {
            "uniform",
            "hedge",
            "adahedge",
            "fixed_share",
            "adaptive_share",
            "directed_fixed_share",
            "ddfs",
        }:
            raise ValueError(f"unknown mode: {mode}")
        self.season = season
        self.config = config or DDFSConfig()
        self.config.validate()
        self.mode = mode
        dimension = 4 + len(self.config.lags)
        self.experts = [
            RecursiveLeastSquares(dimension, factor, self.config.ridge)
            for factor in self.config.forgetting_factors
        ]
        self.weights = np.full(len(self.experts), 1.0 / len(self.experts))
        self.history: list[float] = []
        self.normalized_residuals: list[float] = []
        self.ensemble_errors: list[float] = []
        self.fast_loss = 0.0
        self.slow_loss = 0.0
        self.expert_fast_loss = np.zeros(len(self.experts), dtype=np.float64)
        self.expert_slow_loss = np.zeros(len(self.experts), dtype=np.float64)
        self.cumulative_expert_losses = np.zeros(len(self.experts), dtype=np.float64)
        self.mixability_gap = 0.0
        self._last_features: np.ndarray | None = None
        self._last_prediction: Prediction | None = None
        self._awaiting_update = False

    @property
    def minimum_history(self) -> int:
        return max(self.season + max(self.config.lags), 2 * self.season, 8)

    def ready(self) -> bool:
        return len(self.history) >= self.minimum_history

    def observe_initial(self, value: float) -> None:
        if self._awaiting_update:
            raise RuntimeError("cannot warm-start while an update is pending")
        self._validate_value(value)
        self.history.append(float(value))

    def predict(self) -> Prediction:
        if self._awaiting_update:
            raise RuntimeError("update the previous prediction before predicting again")
        if not self.ready():
            raise RuntimeError(f"need at least {self.minimum_history} observations")
        base = self.history[-self.season]
        scale = self._scale()
        features = self._features()
        residual_predictions = np.array(
            [expert.predict(features) for expert in self.experts], dtype=np.float64
        )
        residual_predictions = np.clip(
            residual_predictions,
            -self.config.prediction_clip,
            self.config.prediction_clip,
        )
        expert_values = base + scale * residual_predictions
        weights = self._prediction_weights()
        value = float(weights @ expert_values)
        radius = self._interval_radius(scale)
        drift_score = self._drift_score()
        share = self._share(drift_score)
        prediction = Prediction(
            value=value,
            lower=value - radius,
            upper=value + radius,
            base=float(base),
            scale=float(scale),
            expert_values=tuple(float(item) for item in expert_values),
            weights=tuple(float(item) for item in weights),
            drift_score=float(drift_score),
            share=float(share),
        )
        self._last_features = features
        self._last_prediction = prediction
        self._awaiting_update = True
        return prediction

    def update(self, outcome: float) -> None:
        if not self._awaiting_update or self._last_prediction is None:
            raise RuntimeError("predict must be called immediately before update")
        self._validate_value(outcome)
        prediction = self._last_prediction
        features = self._last_features
        assert features is not None
        scale = prediction.scale
        expert_values = np.asarray(prediction.expert_values)
        losses = np.minimum(np.abs(expert_values - outcome) / scale, self.config.loss_clip)
        detector_loss = float(np.mean(losses))
        next_fast_loss = (
            (1.0 - self.config.fast_rate) * self.fast_loss
            + self.config.fast_rate * detector_loss
        )
        next_slow_loss = (
            (1.0 - self.config.slow_rate) * self.slow_loss
            + self.config.slow_rate * detector_loss
        )
        next_drift_score = self._score_from_losses(next_fast_loss, next_slow_loss)
        next_expert_fast = (
            (1.0 - self.config.fast_rate) * self.expert_fast_loss
            + self.config.fast_rate * losses
        )
        next_expert_slow = (
            (1.0 - self.config.slow_rate) * self.expert_slow_loss
            + self.config.slow_rate * losses
        )
        if self.mode == "adahedge":
            self._update_adahedge(losses)
        elif self.mode != "uniform":
            posterior = self.weights * np.exp(-self.config.eta * losses)
            total = float(posterior.sum())
            if total <= 0 or not isfinite(total):
                posterior.fill(1.0 / len(posterior))
            else:
                posterior /= total
            if self.mode == "hedge":
                self.weights = posterior
            else:
                share = self._share(next_drift_score)
                target = self._share_target(
                    next_drift_score,
                    expert_fast=next_expert_fast,
                    expert_slow=next_expert_slow,
                )
                self.weights = (1.0 - share) * posterior + share * target
                self.weights /= self.weights.sum()

        normalized_target = (outcome - prediction.base) / scale
        normalized_target = float(
            np.clip(normalized_target, -self.config.prediction_clip, self.config.prediction_clip)
        )
        for expert in self.experts:
            expert.update(features, normalized_target)

        self.fast_loss = next_fast_loss
        self.slow_loss = next_slow_loss
        self.expert_fast_loss = next_expert_fast
        self.expert_slow_loss = next_expert_slow
        self.ensemble_errors.append(abs(prediction.value - outcome))
        residual = (outcome - prediction.base) / scale
        self.normalized_residuals.append(float(np.clip(residual, -20.0, 20.0)))
        self.history.append(float(outcome))
        self._last_features = None
        self._last_prediction = None
        self._awaiting_update = False

    def _features(self) -> np.ndarray:
        scale = self._scale()
        features = [1.0]
        for lag in self.config.lags:
            recent = self.history[-lag]
            seasonal_index = len(self.history) - lag - self.season
            seasonal = self.history[seasonal_index]
            features.append(float(np.clip((recent - seasonal) / scale, -8.0, 8.0)))
        trend = (self.history[-1] - self.history[-2]) / scale
        position = len(self.history) % self.season
        angle = 2.0 * np.pi * position / self.season
        features.extend([float(np.clip(trend, -8.0, 8.0)), float(np.sin(angle)), float(np.cos(angle))])
        return np.asarray(features, dtype=np.float64)

    def _scale(self) -> float:
        window = min(len(self.history) - 1, max(2 * self.season, 32))
        values = np.asarray(self.history[-(window + 1) :], dtype=np.float64)
        differences = np.abs(np.diff(values))
        scale = float(np.median(differences)) * 1.4826
        if not isfinite(scale) or scale < self.config.min_scale:
            scale = float(np.mean(differences)) if differences.size else 1.0
        return max(scale, self.config.min_scale)

    def _prediction_weights(self) -> np.ndarray:
        if self.mode == "uniform":
            return np.full(len(self.experts), 1.0 / len(self.experts))
        return self.weights.copy()

    def _update_adahedge(self, losses: np.ndarray) -> None:
        """Parameter-free AdaHedge update on the shared expert bank.

        The learning rate follows de Rooij et al. (2014),
        ``eta_t = log(K) / Delta_{t-1}``, with the infinite-rate limit handled
        explicitly. Losses are the same causal capped normalized absolute losses
        used by the other aggregation modes.
        """
        count = len(self.experts)
        learner_loss = float(self.weights @ losses)
        if self.mixability_gap <= 0.0:
            mix_loss = float(np.min(losses))
        else:
            eta = np.log(count) / self.mixability_gap
            positive = self.weights > 0.0
            log_terms = np.full(count, -np.inf, dtype=np.float64)
            log_terms[positive] = np.log(self.weights[positive]) - eta * losses[positive]
            maximum = float(np.max(log_terms))
            log_sum = maximum + float(np.log(np.exp(log_terms - maximum).sum()))
            mix_loss = -log_sum / eta
        gap = learner_loss - mix_loss
        self.mixability_gap += max(0.0, gap)
        self.cumulative_expert_losses += losses

        if self.mixability_gap <= 0.0:
            leaders = self.cumulative_expert_losses == np.min(self.cumulative_expert_losses)
            self.weights = leaders.astype(np.float64) / float(np.sum(leaders))
            return
        next_eta = np.log(count) / self.mixability_gap
        logits = -next_eta * self.cumulative_expert_losses
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        self.weights = weights / float(weights.sum())

    def _drift_score(self) -> float:
        return self._score_from_losses(self.fast_loss, self.slow_loss)

    @staticmethod
    def _score_from_losses(fast_loss: float, slow_loss: float) -> float:
        denominator = max(slow_loss, 0.05)
        return float(np.clip((fast_loss - slow_loss) / denominator, 0.0, 1.0))

    def _share(self, drift_score: float) -> float:
        if self.mode in {"hedge", "adahedge", "uniform"}:
            return 0.0
        if self.mode in {"fixed_share", "directed_fixed_share"}:
            return self.config.base_share
        return float(
            np.clip(
                self.config.base_share + self.config.drift_gain * drift_score,
                self.config.base_share,
                self.config.max_share,
            )
        )

    def _share_target(
        self,
        drift_score: float,
        *,
        expert_fast: np.ndarray | None = None,
        expert_slow: np.ndarray | None = None,
    ) -> np.ndarray:
        count = len(self.experts)
        uniform = np.full(count, 1.0 / count)
        if self.mode not in {"directed_fixed_share", "ddfs"} or drift_score <= 0:
            return uniform
        memory_priority = np.asarray(
            [1.0 - factor + 1e-3 for factor in self.config.forgetting_factors],
            dtype=np.float64,
        )
        memory_priority /= memory_priority.sum()
        fast = self.expert_fast_loss if expert_fast is None else expert_fast
        slow = self.expert_slow_loss if expert_slow is None else expert_slow
        relative = fast / np.maximum(slow, 0.05)
        logits = -relative / 0.5
        logits -= float(np.max(logits))
        performance_priority = np.exp(logits)
        performance_priority /= performance_priority.sum()
        directed = np.sqrt(memory_priority * performance_priority)
        directed /= directed.sum()
        return (1.0 - drift_score) * uniform + drift_score * directed

    def _interval_radius(self, scale: float) -> float:
        if len(self.ensemble_errors) < 8:
            return 1.96 * scale
        errors = np.asarray(self.ensemble_errors[-self.config.interval_window :])
        quantile = min(1.0, (1.0 - self.config.interval_alpha) * (len(errors) + 1) / len(errors))
        return max(float(np.quantile(errors, quantile, method="higher")), scale * 0.1)

    @staticmethod
    def _validate_value(value: float) -> None:
        if not isfinite(float(value)):
            raise ValueError("observations must be finite")


class SeasonalNaive:
    def __init__(self, season: int) -> None:
        self.season = season
        self.history: list[float] = []

    def observe_initial(self, value: float) -> None:
        self.history.append(float(value))

    def ready(self) -> bool:
        return len(self.history) >= self.season

    def predict(self) -> float:
        if not self.ready():
            raise RuntimeError("insufficient history")
        return self.history[-self.season]

    def update(self, outcome: float) -> None:
        self.history.append(float(outcome))

