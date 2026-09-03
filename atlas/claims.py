from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .evidence import verify_evidence
from .models import RunManifest


@dataclass(frozen=True)
class ClaimSpec:
    metric_key: str
    operator: str
    threshold: float
    minimum_complete_experiments: int | None = None

    def validate(self) -> None:
        if not self.metric_key.strip():
            raise ValueError("metric_key must be non-empty")
        if self.operator not in {">", ">=", "<", "<=", "=="}:
            raise ValueError(f"unsupported operator: {self.operator}")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.minimum_complete_experiments is not None and self.minimum_complete_experiments < 1:
            raise ValueError("minimum_complete_experiments must be >= 1")


@dataclass
class ClaimReport:
    run_id: str
    classification: str
    evidence_status: str
    metric_key: str
    operator: str
    threshold: float
    aggregate_value: float | None
    values_by_experiment: dict[str, float]
    missing_metric_experiments: list[str]
    malformed_metric_experiments: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "classification": self.classification,
            "evidence_status": self.evidence_status,
            "metric_key": self.metric_key,
            "operator": self.operator,
            "threshold": self.threshold,
            "aggregate_value": self.aggregate_value,
            "values_by_experiment": dict(sorted(self.values_by_experiment.items())),
            "missing_metric_experiments": sorted(self.missing_metric_experiments),
            "malformed_metric_experiments": sorted(self.malformed_metric_experiments),
            "reason": self.reason,
        }


def _compare(value: float, operator: str, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    raise ValueError(f"unsupported operator: {operator}")


def classify_claim(
    manifest: RunManifest,
    evidence_root: str | Path,
    spec: ClaimSpec,
) -> ClaimReport:
    manifest.validate()
    spec.validate()
    root = Path(evidence_root)
    evidence = verify_evidence(manifest, root)

    values: dict[str, float] = {}
    missing_metric: list[str] = []
    malformed_metric: list[str] = []

    for experiment in manifest.experiments:
        metrics_path = root / experiment.name / "metrics.json"
        if not metrics_path.is_file():
            missing_metric.append(experiment.name)
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            malformed_metric.append(experiment.name)
            continue
        if not isinstance(payload, dict) or spec.metric_key not in payload:
            missing_metric.append(experiment.name)
            continue
        raw = payload[spec.metric_key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            malformed_metric.append(experiment.name)
            continue
        value = float(raw)
        if not math.isfinite(value):
            malformed_metric.append(experiment.name)
            continue
        values[experiment.name] = value

    if malformed_metric or evidence.invalid_json:
        return ClaimReport(
            run_id=manifest.run_id,
            classification="INVALID_EVIDENCE",
            evidence_status=evidence.status,
            metric_key=spec.metric_key,
            operator=spec.operator,
            threshold=spec.threshold,
            aggregate_value=None,
            values_by_experiment=values,
            missing_metric_experiments=missing_metric,
            malformed_metric_experiments=malformed_metric,
            reason="One or more preregistered metric artifacts are malformed or invalid JSON.",
        )

    required = spec.minimum_complete_experiments or len(manifest.experiments)
    if len(values) < required or missing_metric:
        aggregate = mean(values.values()) if values else None
        return ClaimReport(
            run_id=manifest.run_id,
            classification="INCONCLUSIVE",
            evidence_status=evidence.status,
            metric_key=spec.metric_key,
            operator=spec.operator,
            threshold=spec.threshold,
            aggregate_value=aggregate,
            values_by_experiment=values,
            missing_metric_experiments=missing_metric,
            malformed_metric_experiments=[],
            reason=(
                f"Only {len(values)} complete metric records are available; "
                f"at least {required} are required and preregistered runs may not be silently dropped."
            ),
        )

    aggregate = mean(values.values())
    supported = _compare(aggregate, spec.operator, spec.threshold)
    return ClaimReport(
        run_id=manifest.run_id,
        classification="SUPPORTED" if supported else "NOT_SUPPORTED",
        evidence_status=evidence.status,
        metric_key=spec.metric_key,
        operator=spec.operator,
        threshold=spec.threshold,
        aggregate_value=aggregate,
        values_by_experiment=values,
        missing_metric_experiments=[],
        malformed_metric_experiments=[],
        reason=(
            f"Mean preregistered {spec.metric_key}={aggregate:.12g} "
            f"{'satisfies' if supported else 'does not satisfy'} {spec.operator} {spec.threshold:.12g}."
        ),
    )
