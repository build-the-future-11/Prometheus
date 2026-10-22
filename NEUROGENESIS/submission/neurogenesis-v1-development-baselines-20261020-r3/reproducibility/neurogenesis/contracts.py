"""Strict data contracts shared by every candidate mechanism and baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping


class ContractError(ValueError):
    """Raised when a protocol or experiment violates a frozen contract."""


class BudgetExceeded(RuntimeError):
    """Raised before work continues beyond an admitted resource budget."""


class ScopeStatus(str, Enum):
    UNRESOLVED = "SCOPE_UNRESOLVED"
    LEARNED_RULE = "APPROVED_LEARNED_RULE"
    MEMORY_WEIGHT = "APPROVED_MEMORY_WEIGHT"


@dataclass(frozen=True)
class ResourceBudget:
    max_steps: int
    max_gradient_evaluations: int
    max_parameter_updates: int
    max_retrievals: int
    max_writes: int
    max_consolidations: int
    max_state_bytes: int
    max_wall_seconds: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ResourceBudget":
        required = {field.name for field in cls.__dataclass_fields__.values()}
        if set(raw) != required:
            missing = sorted(required - set(raw))
            extra = sorted(set(raw) - required)
            raise ContractError(f"resource budget keys mismatch: missing={missing}, extra={extra}")
        budget = cls(**raw)
        for name, value in asdict(budget).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"resource budget {name} must be numeric")
            if not math.isfinite(float(value)) or value < 0:
                raise ContractError(f"resource budget {name} must be finite and non-negative")
        if budget.max_steps < 1 or budget.max_state_bytes < 1 or budget.max_wall_seconds <= 0:
            raise ContractError("step, state-byte, and wall-time budgets must be positive")
        return budget


@dataclass
class ResourceUsage:
    steps: int = 0
    gradient_evaluations: int = 0
    parameter_updates: int = 0
    retrievals: int = 0
    writes: int = 0
    consolidations: int = 0
    peak_state_bytes: int = 0
    wall_seconds: float = 0.0

    def check(self, budget: ResourceBudget) -> None:
        checks = {
            "steps": (self.steps, budget.max_steps),
            "gradient_evaluations": (
                self.gradient_evaluations,
                budget.max_gradient_evaluations,
            ),
            "parameter_updates": (self.parameter_updates, budget.max_parameter_updates),
            "retrievals": (self.retrievals, budget.max_retrievals),
            "writes": (self.writes, budget.max_writes),
            "consolidations": (self.consolidations, budget.max_consolidations),
            "peak_state_bytes": (self.peak_state_bytes, budget.max_state_bytes),
            "wall_seconds": (self.wall_seconds, budget.max_wall_seconds),
        }
        for name, (used, maximum) in checks.items():
            if used > maximum:
                raise BudgetExceeded(f"{name} budget exceeded: used={used}, maximum={maximum}")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TaskIdentity:
    task_id: str
    split: str
    family: str
    seed: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TaskIdentity":
        required = {"task_id", "split", "family", "seed"}
        if set(raw) != required:
            raise ContractError("task identity must contain exactly task_id, split, family, and seed")
        if raw["split"] not in {"meta_train", "development", "evaluation"}:
            raise ContractError(f"unsupported task split: {raw['split']!r}")
        if isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int):
            raise ContractError("task seed must be an integer")
        for name in ("task_id", "family"):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ContractError(f"task identity {name} must be a non-empty string")
        return cls(**raw)


@dataclass(frozen=True)
class Sample:
    step: int
    features: tuple[float, ...]
    target: float
    regime: str
    label_available_at: int

    def __post_init__(self) -> None:
        if self.step < 0 or self.label_available_at <= self.step:
            raise ContractError("labels must become available strictly after prediction time")
        if not self.features or any(not math.isfinite(value) for value in self.features):
            raise ContractError("features must be non-empty and finite")
        if not math.isfinite(self.target):
            raise ContractError("target must be finite")

