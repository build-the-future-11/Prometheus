"""Deterministic sequential task families with explicit information timing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Mapping, Any

from .contracts import ContractError, Sample, TaskIdentity


@dataclass(frozen=True)
class TaskDefinition:
    identity: TaskIdentity
    steps: int
    feature_dim: int
    informative_dim: int
    label_delay: int
    noise_std: float
    regime_length: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TaskDefinition":
        required = {
            "identity",
            "steps",
            "feature_dim",
            "informative_dim",
            "label_delay",
            "noise_std",
            "regime_length",
        }
        if set(raw) != required:
            raise ContractError(
                f"task definition keys mismatch: missing={sorted(required - set(raw))}, "
                f"extra={sorted(set(raw) - required)}"
            )
        definition = cls(
            identity=TaskIdentity.from_mapping(raw["identity"]),
            steps=raw["steps"],
            feature_dim=raw["feature_dim"],
            informative_dim=raw["informative_dim"],
            label_delay=raw["label_delay"],
            noise_std=raw["noise_std"],
            regime_length=raw["regime_length"],
        )
        definition.validate()
        return definition

    def validate(self) -> None:
        integer_fields = {
            "steps": self.steps,
            "feature_dim": self.feature_dim,
            "informative_dim": self.informative_dim,
            "label_delay": self.label_delay,
            "regime_length": self.regime_length,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"{name} must be a positive integer")
        if self.informative_dim >= self.feature_dim:
            raise ContractError("feature_dim must leave at least one distractor dimension")
        if self.steps != self.regime_length * 4:
            raise ContractError("the frozen task has exactly four equal-length regimes")
        if not isinstance(self.noise_std, (int, float)) or self.noise_std < 0:
            raise ContractError("noise_std must be non-negative")


def _unit_vector(rng: random.Random, dimension: int) -> tuple[float, ...]:
    raw = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
    norm = math.sqrt(sum(value * value for value in raw))
    return tuple(value / norm for value in raw)


def generate_task(definition: TaskDefinition) -> list[Sample]:
    """Generate A -> conflicting B -> returning A -> novel C observations."""

    definition.validate()
    rng = random.Random(definition.identity.seed)
    weights_a = _unit_vector(rng, definition.informative_dim)
    weights_b = tuple(-value for value in weights_a)
    raw_c = _unit_vector(rng, definition.informative_dim)
    projection = sum(a * c for a, c in zip(weights_a, raw_c))
    orthogonal = [c - projection * a for a, c in zip(weights_a, raw_c)]
    norm_c = math.sqrt(sum(value * value for value in orthogonal))
    if norm_c < 1e-12:
        orthogonal = list(reversed(weights_a))
        norm_c = math.sqrt(sum(value * value for value in orthogonal))
    weights_c = tuple(value / norm_c for value in orthogonal)
    regimes = (
        ("A_initial", weights_a),
        ("B_conflict", weights_b),
        ("A_return", weights_a),
        ("C_novel", weights_c),
    )

    samples: list[Sample] = []
    for step in range(definition.steps):
        regime_index = min(step // definition.regime_length, 3)
        regime, weights = regimes[regime_index]
        informative = [rng.gauss(0.0, 1.0) for _ in range(definition.informative_dim)]
        # High-variance distractors change distribution but never enter the target.
        distractor_scale = 3.0 if regime in {"B_conflict", "A_return"} else 1.0
        distractors = [
            rng.gauss(0.0, distractor_scale)
            for _ in range(definition.feature_dim - definition.informative_dim)
        ]
        target = sum(weight * value for weight, value in zip(weights, informative))
        target += rng.gauss(0.0, float(definition.noise_std))
        samples.append(
            Sample(
                step=step,
                features=tuple(informative + distractors),
                target=target,
                regime=regime,
                label_available_at=step + definition.label_delay,
            )
        )
    return samples


def validate_identity_sets(identity_groups: Iterable[Iterable[TaskIdentity]]) -> None:
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, int]] = set()
    for group in identity_groups:
        for identity in group:
            if identity.task_id in seen_ids:
                raise ContractError(f"duplicate task_id across splits: {identity.task_id}")
            pair = (identity.family, identity.seed)
            if pair in seen_pairs:
                raise ContractError(f"duplicate family/seed across splits: {pair}")
            seen_ids.add(identity.task_id)
            seen_pairs.add(pair)

