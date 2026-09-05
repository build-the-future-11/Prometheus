from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from .evidence import verify_evidence
from .models import RunManifest


_ALLOWED_OPERATORS = {">", ">=", "<", "<=", "=="}
_ALLOWED_AGGREGATIONS = {"mean", "median", "min", "max"}
_FAILURE_STATUSES = {"FAILED", "ERROR", "CANCELLED", "CANCELED", "TIMEOUT", "BLOCKED"}


@dataclass(frozen=True)
class ClaimSpec:
    metric_key: str
    operator: str
    threshold: float
    tolerance: float
    aggregation_rule: str
    experiment_ids: tuple[str, ...]

    @property
    def direction(self) -> str:
        if self.operator in {">", ">="}:
            return "higher"
        if self.operator in {"<", "<="}:
            return "lower"
        return "target"

    def validate(self) -> None:
        if not self.metric_key.strip():
            raise ValueError("metric_key must be non-empty")
        if self.operator not in _ALLOWED_OPERATORS:
            raise ValueError(f"unsupported operator: {self.operator}")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        if self.aggregation_rule not in _ALLOWED_AGGREGATIONS:
            raise ValueError(f"unsupported aggregation_rule: {self.aggregation_rule}")
        if not self.experiment_ids:
            raise ValueError("experiment_ids must freeze at least one preregistered experiment")
        if any(not item.strip() for item in self.experiment_ids):
            raise ValueError("experiment_ids must be non-empty strings")
        if len(set(self.experiment_ids)) != len(self.experiment_ids):
            raise ValueError("experiment_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "metric_key": self.metric_key,
            "direction": self.direction,
            "operator": self.operator,
            "threshold": self.threshold,
            "tolerance": self.tolerance,
            "aggregation_rule": self.aggregation_rule,
            "experiment_ids": list(self.experiment_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimSpec":
        if not isinstance(payload, dict):
            raise ValueError("claim spec must be a JSON object")
        required = {
            "metric_key",
            "operator",
            "threshold",
            "tolerance",
            "aggregation_rule",
            "experiment_ids",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"claim spec missing fields: {', '.join(missing)}")
        raw_ids = payload["experiment_ids"]
        if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
            raise ValueError("claim spec experiment_ids must be a JSON array of strings")
        threshold = payload["threshold"]
        tolerance = payload["tolerance"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("claim spec threshold must be numeric")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ValueError("claim spec tolerance must be numeric")
        spec = cls(
            metric_key=str(payload["metric_key"]),
            operator=str(payload["operator"]),
            threshold=float(threshold),
            tolerance=float(tolerance),
            aggregation_rule=str(payload["aggregation_rule"]),
            experiment_ids=tuple(raw_ids),
        )
        spec.validate()
        if "direction" in payload and str(payload["direction"]) != spec.direction:
            raise ValueError("claim spec direction does not match operator")
        return spec


@dataclass
class ClaimReport:
    run_id: str
    classification: str
    evidence_status: str
    metric_key: str
    direction: str
    operator: str
    threshold: float
    tolerance: float
    aggregation_rule: str
    experiment_ids: tuple[str, ...]
    aggregate_value: float | None
    values_by_experiment: dict[str, float]
    missing_metric_experiments: list[str]
    malformed_metric_experiments: list[str]
    failed_experiments: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "classification": self.classification,
            "evidence_status": self.evidence_status,
            "metric_key": self.metric_key,
            "direction": self.direction,
            "operator": self.operator,
            "threshold": self.threshold,
            "tolerance": self.tolerance,
            "aggregation_rule": self.aggregation_rule,
            "experiment_ids": list(self.experiment_ids),
            "aggregate_value": self.aggregate_value,
            "values_by_experiment": dict(sorted(self.values_by_experiment.items())),
            "missing_metric_experiments": sorted(self.missing_metric_experiments),
            "malformed_metric_experiments": sorted(self.malformed_metric_experiments),
            "failed_experiments": sorted(self.failed_experiments),
            "reason": self.reason,
        }


def load_claim_spec(path: str | Path) -> ClaimSpec:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid claim spec {source}: {exc}") from exc
    return ClaimSpec.from_dict(payload)


def _compare(value: float, operator: str, threshold: float, tolerance: float) -> bool:
    if operator == ">":
        return value > threshold - tolerance
    if operator == ">=":
        return value >= threshold - tolerance
    if operator == "<":
        return value < threshold + tolerance
    if operator == "<=":
        return value <= threshold + tolerance
    if operator == "==":
        return abs(value - threshold) <= tolerance
    raise ValueError(f"unsupported operator: {operator}")


def _aggregate(values: list[float], rule: str) -> float:
    if rule == "mean":
        return mean(values)
    if rule == "median":
        return median(values)
    if rule == "min":
        return min(values)
    if rule == "max":
        return max(values)
    raise ValueError(f"unsupported aggregation_rule: {rule}")


def _report(
    manifest: RunManifest,
    evidence_status: str,
    spec: ClaimSpec,
    *,
    classification: str,
    aggregate_value: float | None,
    values: dict[str, float],
    missing_metric: list[str],
    malformed_metric: list[str],
    failed: list[str],
    reason: str,
) -> ClaimReport:
    return ClaimReport(
        run_id=manifest.run_id,
        classification=classification,
        evidence_status=evidence_status,
        metric_key=spec.metric_key,
        direction=spec.direction,
        operator=spec.operator,
        threshold=spec.threshold,
        tolerance=spec.tolerance,
        aggregation_rule=spec.aggregation_rule,
        experiment_ids=spec.experiment_ids,
        aggregate_value=aggregate_value,
        values_by_experiment=values,
        missing_metric_experiments=missing_metric,
        malformed_metric_experiments=malformed_metric,
        failed_experiments=failed,
        reason=reason,
    )


def classify_claim(
    manifest: RunManifest,
    evidence_root: str | Path,
    spec: ClaimSpec,
) -> ClaimReport:
    manifest.validate()
    spec.validate()
    manifest_ids = tuple(experiment.name for experiment in manifest.experiments)
    if set(spec.experiment_ids) != set(manifest_ids):
        missing = sorted(set(manifest_ids) - set(spec.experiment_ids))
        unknown = sorted(set(spec.experiment_ids) - set(manifest_ids))
        details: list[str] = []
        if missing:
            details.append(f"omitted manifest experiments: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown experiment IDs: {', '.join(unknown)}")
        raise ValueError("claim spec experiment_ids must exactly freeze the manifest experiment set; " + "; ".join(details))

    root = Path(evidence_root)
    evidence = verify_evidence(manifest, root)
    selected = set(spec.experiment_ids)

    values: dict[str, float] = {}
    missing_metric: list[str] = []
    malformed_metric: list[str] = []
    failed: list[str] = []

    for experiment in manifest.experiments:
        if experiment.name not in selected:
            continue
        metrics_path = root / experiment.name / "metrics.json"
        if not metrics_path.is_file():
            missing_metric.append(experiment.name)
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            malformed_metric.append(experiment.name)
            continue
        if not isinstance(payload, dict):
            malformed_metric.append(experiment.name)
            continue

        raw_status = payload.get("status")
        if isinstance(raw_status, str) and raw_status.strip().upper() in _FAILURE_STATUSES:
            failed.append(experiment.name)

        if spec.metric_key not in payload:
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

    ordered_values = [values[experiment_id] for experiment_id in spec.experiment_ids if experiment_id in values]
    aggregate = _aggregate(ordered_values, spec.aggregation_rule) if ordered_values else None

    if malformed_metric or evidence.invalid_json:
        return _report(
            manifest,
            evidence.status,
            spec,
            classification="INVALID_EVIDENCE",
            aggregate_value=aggregate,
            values=values,
            missing_metric=missing_metric,
            malformed_metric=malformed_metric,
            failed=failed,
            reason="One or more preregistered metric artifacts are malformed or invalid JSON.",
        )

    if evidence.status != "COMPLETE":
        return _report(
            manifest,
            evidence.status,
            spec,
            classification="INCONCLUSIVE",
            aggregate_value=aggregate,
            values=values,
            missing_metric=missing_metric,
            malformed_metric=[],
            failed=failed,
            reason="Required preregistered evidence is incomplete; incomplete runs may not be converted into scientific support.",
        )

    if failed:
        return _report(
            manifest,
            evidence.status,
            spec,
            classification="INCONCLUSIVE",
            aggregate_value=aggregate,
            values=values,
            missing_metric=missing_metric,
            malformed_metric=[],
            failed=failed,
            reason="One or more preregistered experiments explicitly report a failure terminal status.",
        )

    if missing_metric:
        return _report(
            manifest,
            evidence.status,
            spec,
            classification="INCONCLUSIVE",
            aggregate_value=aggregate,
            values=values,
            missing_metric=missing_metric,
            malformed_metric=[],
            failed=[],
            reason="One or more preregistered experiments lack the frozen claim metric.",
        )

    if aggregate is None:
        return _report(
            manifest,
            evidence.status,
            spec,
            classification="INCONCLUSIVE",
            aggregate_value=None,
            values=values,
            missing_metric=missing_metric,
            malformed_metric=[],
            failed=[],
            reason="No complete preregistered metric values are available.",
        )

    supported = _compare(aggregate, spec.operator, spec.threshold, spec.tolerance)
    tolerance_text = f" with tolerance {spec.tolerance:.12g}" if spec.tolerance else ""
    return _report(
        manifest,
        evidence.status,
        spec,
        classification="SUPPORTED" if supported else "NOT_SUPPORTED",
        aggregate_value=aggregate,
        values=values,
        missing_metric=[],
        malformed_metric=[],
        failed=[],
        reason=(
            f"{spec.aggregation_rule} preregistered {spec.metric_key}={aggregate:.12g} "
            f"{'satisfies' if supported else 'does not satisfy'} {spec.operator} {spec.threshold:.12g}"
            f"{tolerance_text}."
        ),
    )
