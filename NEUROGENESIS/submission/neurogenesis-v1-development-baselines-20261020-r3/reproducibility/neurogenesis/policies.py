"""Conventional dependency-free linear baselines and checkpoint handling."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ContractError, ResourceUsage


def squared_error_gradient(
    features: Sequence[float],
    target: float,
    prediction: float,
) -> tuple[list[float], float]:
    """Gradient of 0.5 * (prediction - target)^2 for a linear model."""

    error = prediction - target
    return [error * value for value in features], error


class LinearPolicy(ABC):
    """Common policy API. Targets are passed only to observe_available_label."""

    policy_type = "abstract"

    def __init__(self, feature_dim: int) -> None:
        if feature_dim < 1:
            raise ContractError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.weights = [0.0] * feature_dim
        self.bias = 0.0
        self.usage = ResourceUsage()
        self.max_update_norm = 0.0

    def predict(self, features: Sequence[float]) -> float:
        self._validate_features(features)
        prediction = sum(weight * value for weight, value in zip(self.weights, features)) + self.bias
        if not math.isfinite(prediction):
            raise FloatingPointError("non-finite prediction")
        return prediction

    def _validate_features(self, features: Sequence[float]) -> None:
        if len(features) != self.feature_dim:
            raise ContractError(
                f"feature dimension mismatch: expected={self.feature_dim}, received={len(features)}"
            )
        if any(not math.isfinite(float(value)) for value in features):
            raise ContractError("features must be finite")

    @abstractmethod
    def observe_available_label(
        self,
        features: Sequence[float],
        target: float,
        prediction_at_observation: float,
    ) -> None:
        """Consume a label only after the runner has established availability."""

    @abstractmethod
    def optimizer_state(self) -> Mapping[str, Any]:
        """Return JSON-safe optimizer state."""

    @abstractmethod
    def load_optimizer_state(self, state: Mapping[str, Any]) -> None:
        """Restore optimizer state after strict validation."""

    def state_bytes(self) -> int:
        # Float64-equivalent algorithmic state, independent of Python object overhead.
        return 8 * (self.feature_dim + 1)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "neurogenesis.linear-policy.v1",
            "policy_type": self.policy_type,
            "feature_dim": self.feature_dim,
            "weights": list(self.weights),
            "bias": self.bias,
            "usage": self.usage.to_dict(),
            "max_update_norm": self.max_update_norm,
            "optimizer_state": dict(self.optimizer_state()),
        }

    def restore_common(self, snapshot: Mapping[str, Any]) -> None:
        required = {
            "schema_version",
            "policy_type",
            "feature_dim",
            "weights",
            "bias",
            "usage",
            "max_update_norm",
            "optimizer_state",
        }
        if set(snapshot) != required:
            raise ContractError("checkpoint fields do not match the v1 policy schema")
        if snapshot["schema_version"] != "neurogenesis.linear-policy.v1":
            raise ContractError("unsupported checkpoint schema")
        if snapshot["policy_type"] != self.policy_type:
            raise ContractError("checkpoint policy type mismatch")
        if snapshot["feature_dim"] != self.feature_dim:
            raise ContractError("checkpoint feature dimension mismatch")
        weights = snapshot["weights"]
        if not isinstance(weights, list) or len(weights) != self.feature_dim:
            raise ContractError("checkpoint weights malformed")
        values = [float(value) for value in weights]
        bias = float(snapshot["bias"])
        if any(not math.isfinite(value) for value in values + [bias]):
            raise ContractError("checkpoint parameters must be finite")
        usage_raw = snapshot["usage"]
        if set(usage_raw) != set(ResourceUsage().__dict__):
            raise ContractError("checkpoint usage fields malformed")
        self.weights = values
        self.bias = bias
        self.usage = ResourceUsage(**usage_raw)
        self.max_update_norm = float(snapshot["max_update_norm"])
        self.load_optimizer_state(snapshot["optimizer_state"])


class FrozenLinearPolicy(LinearPolicy):
    policy_type = "no_adaptation"

    def observe_available_label(
        self,
        features: Sequence[float],
        target: float,
        prediction_at_observation: float,
    ) -> None:
        self._validate_features(features)
        if not math.isfinite(target) or not math.isfinite(prediction_at_observation):
            raise ContractError("label and prediction must be finite")

    def optimizer_state(self) -> Mapping[str, Any]:
        return {}

    def load_optimizer_state(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ContractError("frozen policy checkpoint cannot contain optimizer state")


class SGDLinearPolicy(LinearPolicy):
    policy_type = "standard_sgd"

    def __init__(
        self,
        feature_dim: int,
        learning_rate: float,
        gradient_clip: float = 10.0,
    ) -> None:
        super().__init__(feature_dim)
        if not (0 < learning_rate < 1) or gradient_clip <= 0:
            raise ContractError("SGD learning_rate and gradient_clip must be positive and bounded")
        self.learning_rate = float(learning_rate)
        self.gradient_clip = float(gradient_clip)

    def observe_available_label(
        self,
        features: Sequence[float],
        target: float,
        prediction_at_observation: float,
    ) -> None:
        self._validate_features(features)
        gradients, bias_gradient = squared_error_gradient(features, target, prediction_at_observation)
        norm = math.sqrt(sum(value * value for value in gradients) + bias_gradient**2)
        scale = min(1.0, self.gradient_clip / max(norm, 1e-30))
        updates = [-self.learning_rate * scale * value for value in gradients]
        bias_update = -self.learning_rate * scale * bias_gradient
        self.weights = [weight + update for weight, update in zip(self.weights, updates)]
        self.bias += bias_update
        update_norm = math.sqrt(sum(value * value for value in updates) + bias_update**2)
        self.max_update_norm = max(self.max_update_norm, update_norm)
        self.usage.gradient_evaluations += 1
        self.usage.parameter_updates += 1
        if any(not math.isfinite(value) for value in self.weights + [self.bias]):
            raise FloatingPointError("SGD produced non-finite parameters")

    def optimizer_state(self) -> Mapping[str, Any]:
        return {"learning_rate": self.learning_rate, "gradient_clip": self.gradient_clip}

    def load_optimizer_state(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"learning_rate", "gradient_clip"}:
            raise ContractError("SGD optimizer state malformed")
        if float(state["learning_rate"]) != self.learning_rate:
            raise ContractError("SGD checkpoint learning rate mismatch")
        if float(state["gradient_clip"]) != self.gradient_clip:
            raise ContractError("SGD checkpoint gradient clip mismatch")


class AdamLinearPolicy(LinearPolicy):
    policy_type = "mechanism_off_standard_adam"

    def __init__(
        self,
        feature_dim: int,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        gradient_clip: float = 10.0,
    ) -> None:
        super().__init__(feature_dim)
        if not (0 < learning_rate < 1):
            raise ContractError("Adam learning rate must be in (0, 1)")
        if not (0 < beta1 < 1 and 0 < beta2 < 1 and epsilon > 0 and gradient_clip > 0):
            raise ContractError("Adam hyperparameters are invalid")
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.gradient_clip = float(gradient_clip)
        self.first = [0.0] * (feature_dim + 1)
        self.second = [0.0] * (feature_dim + 1)
        self.time = 0

    def observe_available_label(
        self,
        features: Sequence[float],
        target: float,
        prediction_at_observation: float,
    ) -> None:
        self._validate_features(features)
        gradients, bias_gradient = squared_error_gradient(features, target, prediction_at_observation)
        gradient = gradients + [bias_gradient]
        norm = math.sqrt(sum(value * value for value in gradient))
        scale = min(1.0, self.gradient_clip / max(norm, 1e-30))
        gradient = [value * scale for value in gradient]
        self.time += 1
        updates: list[float] = []
        for index, value in enumerate(gradient):
            self.first[index] = self.beta1 * self.first[index] + (1 - self.beta1) * value
            self.second[index] = self.beta2 * self.second[index] + (1 - self.beta2) * value * value
            corrected_first = self.first[index] / (1 - self.beta1**self.time)
            corrected_second = self.second[index] / (1 - self.beta2**self.time)
            updates.append(
                -self.learning_rate
                * corrected_first
                / (math.sqrt(corrected_second) + self.epsilon)
            )
        self.weights = [weight + update for weight, update in zip(self.weights, updates[:-1])]
        self.bias += updates[-1]
        self.max_update_norm = max(
            self.max_update_norm,
            math.sqrt(sum(value * value for value in updates)),
        )
        self.usage.gradient_evaluations += 1
        self.usage.parameter_updates += 1
        if any(not math.isfinite(value) for value in self.weights + [self.bias]):
            raise FloatingPointError("Adam produced non-finite parameters")

    def state_bytes(self) -> int:
        return super().state_bytes() + 8 * 2 * (self.feature_dim + 1) + 8

    def optimizer_state(self) -> Mapping[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "gradient_clip": self.gradient_clip,
            "first": list(self.first),
            "second": list(self.second),
            "time": self.time,
        }

    def load_optimizer_state(self, state: Mapping[str, Any]) -> None:
        required = {
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "gradient_clip",
            "first",
            "second",
            "time",
        }
        if set(state) != required:
            raise ContractError("Adam optimizer state malformed")
        expected = {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "gradient_clip": self.gradient_clip,
        }
        for name, expected_value in expected.items():
            if float(state[name]) != expected_value:
                raise ContractError(f"Adam checkpoint {name} mismatch")
        for name in ("first", "second"):
            values = state[name]
            if not isinstance(values, list) or len(values) != self.feature_dim + 1:
                raise ContractError(f"Adam checkpoint {name} malformed")
            setattr(self, name, [float(value) for value in values])
        self.time = int(state["time"])


def build_policy(raw: Mapping[str, Any], feature_dim: int) -> LinearPolicy:
    required = {"name", "kind", "hyperparameters"}
    if set(raw) != required:
        raise ContractError("policy definition must contain exactly name, kind, and hyperparameters")
    kind = raw["kind"]
    hyperparameters = raw["hyperparameters"]
    if kind == "no_adaptation":
        if hyperparameters:
            raise ContractError("no-adaptation baseline cannot have hyperparameters")
        return FrozenLinearPolicy(feature_dim)
    if kind == "standard_sgd":
        return SGDLinearPolicy(feature_dim=feature_dim, **hyperparameters)
    if kind == "mechanism_off_standard_adam":
        return AdamLinearPolicy(feature_dim=feature_dim, **hyperparameters)
    raise ContractError(f"unknown or unapproved policy kind: {kind!r}")


def save_checkpoint(policy: LinearPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(policy.snapshot(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_checkpoint(policy: LinearPolicy, path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContractError("checkpoint must be a JSON object")
    policy.restore_common(raw)

