"""Grouped linear model with strict shape and serialization contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .linalg import dot


@dataclass
class GroupedLinearModel:
    dimension: int
    groups: list[list[int]]
    weights: list[float]
    intercept: float = 0.0
    fit_intercept: bool = False

    @classmethod
    def zeros(cls, dimension: int, groups: list[list[int]], *, fit_intercept: bool = False) -> "GroupedLinearModel":
        model = cls(
            dimension=dimension,
            groups=[list(group) for group in groups],
            weights=[0.0] * dimension,
            intercept=0.0,
            fit_intercept=fit_intercept,
        )
        model.validate()
        return model

    def validate(self) -> None:
        if self.dimension <= 0 or len(self.weights) != self.dimension:
            raise ValueError("model dimension and weight shape disagree")
        flattened = [index for group in self.groups for index in group]
        if sorted(flattened) != list(range(self.dimension)) or len(set(flattened)) != self.dimension:
            raise ValueError("groups must partition every feature exactly once")
        if not all(math.isfinite(value) for value in self.weights) or not math.isfinite(self.intercept):
            raise ValueError("model parameters must be finite")

    def predict_row(self, row: list[float]) -> float:
        if len(row) != self.dimension:
            raise ValueError(f"feature row has width {len(row)}; expected {self.dimension}")
        return dot(row, self.weights) + self.intercept

    def predict(self, rows: list[list[float]]) -> list[float]:
        return [self.predict_row(row) for row in rows]

    def group_prediction(self, rows: list[list[float]], group: list[int]) -> list[float]:
        return [math.fsum(row[index] * self.weights[index] for index in group) for row in rows]

    def set_group_weights(self, group: list[int], values: list[float]) -> None:
        if len(group) != len(values):
            raise ValueError("group/value shape mismatch")
        candidate = list(self.weights)
        for index, value in zip(group, values, strict=True):
            if not math.isfinite(value):
                raise ValueError("refusing non-finite model update")
            candidate[index] = value
        self.weights = candidate
        self.validate()

    def set_intercept(self, value: float) -> None:
        if not self.fit_intercept:
            if value != 0.0:
                raise ValueError("cannot set an intercept when fit_intercept is disabled")
            return
        if not math.isfinite(value):
            raise ValueError("refusing non-finite intercept")
        self.intercept = value
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "derived-learning.grouped-linear-model.v2",
            "dimension": self.dimension,
            "groups": self.groups,
            "weights": self.weights,
            "intercept": self.intercept,
            "fit_intercept": self.fit_intercept,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroupedLinearModel":
        if payload.get("schema_version") not in {"derived-learning.grouped-linear-model.v1", "derived-learning.grouped-linear-model.v2"}:
            raise ValueError("unsupported model schema")
        model = cls(
            dimension=int(payload["dimension"]),
            groups=[[int(index) for index in group] for group in payload["groups"]],
            weights=[float(value) for value in payload["weights"]],
            intercept=float(payload.get("intercept", 0.0)),
            fit_intercept=bool(payload.get("fit_intercept", False)),
        )
        model.validate()
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GroupedLinearModel":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

