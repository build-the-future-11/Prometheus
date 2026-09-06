from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Criterion:
    metric: str
    operator: str
    threshold: float
    rationale: str

    def validate(self) -> None:
        if self.operator not in {">", ">=", "<", "<=", "=="}:
            raise ValueError(f"unsupported operator: {self.operator}")
        if not self.metric.strip():
            raise ValueError("criterion metric must be non-empty")
        if not math.isfinite(self.threshold):
            raise ValueError("criterion threshold must be finite")
        if not self.rationale.strip():
            raise ValueError("criterion rationale must be non-empty")


@dataclass(frozen=True)
class Experiment:
    name: str
    treatment: str
    control: str
    seed: int
    perturbation: str
    required_artifacts: tuple[str, ...]

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name must be non-empty")
        if not self.treatment.strip() or not self.control.strip():
            raise ValueError("experiment treatment and control must be non-empty")
        if self.treatment.strip() == self.control.strip():
            raise ValueError("treatment and control must differ")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.perturbation.strip():
            raise ValueError("experiment perturbation must be non-empty")
        if not self.required_artifacts:
            raise ValueError("experiment must require at least one artifact")
        if any(not isinstance(artifact, str) or not artifact.strip() for artifact in self.required_artifacts):
            raise ValueError("required artifact paths must be non-empty strings")
        if len(set(self.required_artifacts)) != len(self.required_artifacts):
            raise ValueError("required artifact paths must be unique")


@dataclass
class RunManifest:
    run_id: str
    question: str
    hypothesis: str
    null_hypothesis: str
    primary_criterion: Criterion
    experiments: list[Experiment]
    stopping_rule: str
    evidence_requirements: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if len(self.question.strip()) < 10:
            raise ValueError("research question is too short")
        if not self.hypothesis.strip() or not self.null_hypothesis.strip():
            raise ValueError("hypotheses must be non-empty")
        self.primary_criterion.validate()
        if not self.experiments:
            raise ValueError("manifest must contain experiments")
        for experiment in self.experiments:
            experiment.validate()
        if len({e.name for e in self.experiments}) != len(self.experiments):
            raise ValueError("experiment names must be unique")
        if not self.stopping_rule.strip():
            raise ValueError("stopping rule must be non-empty")
        if not self.evidence_requirements:
            raise ValueError("at least one evidence requirement is required")
        if any(
            not isinstance(requirement, str) or not requirement.strip()
            for requirement in self.evidence_requirements
        ):
            raise ValueError("evidence requirements must be non-empty strings")
        if len(set(self.evidence_requirements)) != len(self.evidence_requirements):
            raise ValueError("evidence requirements must be unique")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
