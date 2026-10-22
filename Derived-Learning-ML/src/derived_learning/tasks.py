"""Deterministic task-family generation with explicit split boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskFamily:
    name: str
    dimension: int
    group_size: int
    train_examples: int
    evaluation_examples: int
    feature_correlation: float
    coefficient_scale: float
    noise_std: float
    interaction_strength: float = 0.0
    nonlinear_kind: str = "linear"
    supplied_partition: str = "true"

    def validate(self) -> None:
        if self.dimension <= 0 or self.group_size <= 0 or self.dimension % self.group_size:
            raise ValueError("dimension must be positive and divisible by group_size")
        if self.train_examples <= 0 or self.evaluation_examples <= 0:
            raise ValueError("task families require positive train and evaluation sizes")
        if not 0.0 <= self.feature_correlation < 1.0:
            raise ValueError("feature_correlation must be in [0, 1)")
        if self.coefficient_scale <= 0.0 or self.noise_std < 0.0 or self.interaction_strength < 0.0:
            raise ValueError("coefficient_scale must be positive; noise and interaction strength must be non-negative")
        if self.nonlinear_kind not in {"linear", "quadratic", "sinusoidal"}:
            raise ValueError("nonlinear_kind must be linear, quadratic, or sinusoidal")
        if self.supplied_partition not in {"true", "interleaved"}:
            raise ValueError("supplied_partition must be true or interleaved")
        if self.supplied_partition == "interleaved" and self.group_size != 2:
            raise ValueError("interleaved partition currently requires group_size=2")


@dataclass(frozen=True)
class Dataset:
    features: list[list[float]]
    targets: list[float]


@dataclass(frozen=True)
class LearningTask:
    task_id: str
    split: str
    family: str
    seed: int
    groups: list[list[int]]
    support: Dataset
    evaluation: Dataset
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_example(rng: random.Random, family: TaskFamily) -> list[float]:
    common = rng.gauss(0.0, 1.0)
    shared = math.sqrt(family.feature_correlation) * common
    independent_scale = math.sqrt(1.0 - family.feature_correlation)
    return [shared + independent_scale * rng.gauss(0.0, 1.0) for _ in range(family.dimension)]


def _sample_coefficients(rng: random.Random, family: TaskFamily) -> list[float]:
    # Alternating signs prevent the shared correlated direction from dominating.
    return [
        family.coefficient_scale * (1.0 if index % 2 == 0 else -1.0) * rng.uniform(0.35, 1.0)
        for index in range(family.dimension)
    ]


def _feature_basis(value: float, nonlinear_kind: str) -> float:
    if nonlinear_kind == "linear":
        return value
    if nonlinear_kind == "quadratic":
        return value + 0.25 * value * value
    if nonlinear_kind == "sinusoidal":
        return math.sin(value)
    raise AssertionError("TaskFamily validation should reject this nonlinear_kind")


def _supplied_groups(family: TaskFamily) -> list[list[int]]:
    true_groups = [
        list(range(start, start + family.group_size))
        for start in range(0, family.dimension, family.group_size)
    ]
    if family.supplied_partition == "true":
        return true_groups
    # Every feature remains in exactly one learner-visible block, but each block
    # mixes features from adjacent generating components.  This is a controlled
    # partition error rather than a hidden-label leak.
    return [
        [2 * index, 2 * ((index + 1) % (family.dimension // 2)) + 1]
        for index in range(family.dimension // 2)
    ]


def _target(row: list[float], coefficients: list[float], family: TaskFamily) -> float:
    additive = math.fsum(
        coefficient * _feature_basis(value, family.nonlinear_kind)
        for value, coefficient in zip(row, coefficients, strict=True)
    )
    # Pair neighbouring true components.  The coefficients and component values
    # intentionally stay generator-internal; the learner receives only x, y,
    # and the possibly misspecified supplied partition.
    interactions = math.fsum(
        family.interaction_strength * coefficients[index] * coefficients[index + 1] * row[index] * row[index + 1]
        for index in range(0, family.dimension, family.group_size)
    )
    return additive + interactions


def generate_task(split: str, family: TaskFamily, seed: int) -> LearningTask:
    """Generate support and sealed evaluation examples from independent RNG streams."""

    family.validate()
    coefficient_rng = random.Random(stable_seed("coefficients", split, family.name, seed))
    coefficients = _sample_coefficients(coefficient_rng, family)
    groups = _supplied_groups(family)

    def dataset(label: str, count: int) -> Dataset:
        rng = random.Random(stable_seed("examples", split, family.name, seed, label))
        features: list[list[float]] = []
        targets: list[float] = []
        for _ in range(count):
            row = _sample_example(rng, family)
            noise = rng.gauss(0.0, family.noise_std)
            target = _target(row, coefficients, family) + noise
            features.append(row)
            targets.append(target)
        return Dataset(features=features, targets=targets)

    task_id = f"{split}-{family.name}-s{seed}"
    return LearningTask(
        task_id=task_id,
        split=split,
        family=family.name,
        seed=seed,
        groups=groups,
        support=dataset("support", family.train_examples),
        evaluation=dataset("evaluation", family.evaluation_examples),
        provenance={
            "generator": "derived_learning.tasks.generate_task.v1",
            "coefficient_labels_exposed_to_learner": False,
            "true_partition_exposed_to_learner": family.supplied_partition == "true",
            "evaluation_targets_exposed_during_fit": False,
            "task_seed_derivation": "sha256(split|family|seed|stream)",
            "structural_regime": {
                "interaction_strength": family.interaction_strength,
                "nonlinear_kind": family.nonlinear_kind,
                "supplied_partition": family.supplied_partition,
            },
        },
    )


def family_from_dict(payload: dict[str, Any]) -> TaskFamily:
    family = TaskFamily(**payload)
    family.validate()
    return family


def generate_from_config(config: dict[str, Any]) -> list[LearningTask]:
    tasks: list[LearningTask] = []
    for split in ("meta_train", "development", "final", "long_horizon"):
        split_config = config["splits"].get(split)
        if split_config is None:
            continue
        family = family_from_dict(split_config["family"])
        for seed in split_config["seeds"]:
            tasks.append(generate_task(split, family, int(seed)))
    return tasks


def write_tasks(tasks: list[LearningTask], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "derived-learning.tasks.v1",
        "tasks": [task.to_dict() for task in tasks],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

