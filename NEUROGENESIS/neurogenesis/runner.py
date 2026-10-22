"""Serial prequential execution with delayed labels and retained attempt records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

from .contracts import ContractError, ResourceBudget
from .policies import LinearPolicy, build_policy, save_checkpoint
from .protocol import Protocol, canonical_json_bytes, sha256_json, strict_json_load
from .tasks import TaskDefinition, generate_task, validate_identity_sets


@dataclass(frozen=True)
class PendingLabel:
    source_step: int
    available_at: int
    features: tuple[float, ...]
    target: float
    prediction: float


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _window_mse(trace: list[dict[str, Any]], start: int, stop: int) -> float:
    return _mean(
        [row["squared_error"] for row in trace if start <= row["step"] < stop]
    )


def execute_task(
    definition: TaskDefinition,
    policy: LinearPolicy,
    budget: ResourceBudget,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = generate_task(definition)
    if len(samples) > budget.max_steps:
        raise ContractError("task length exceeds admitted step budget")
    start_time = time.perf_counter()
    pending: dict[int, list[PendingLabel]] = defaultdict(list)
    trace: list[dict[str, Any]] = []
    nonfinite_predictions = 0

    for sample in samples:
        # Only labels whose registered availability time has arrived may update state.
        for label in pending.pop(sample.step, []):
            policy.observe_available_label(
                label.features,
                label.target,
                label.prediction,
            )
        policy.usage.peak_state_bytes = max(policy.usage.peak_state_bytes, policy.state_bytes())
        policy.usage.wall_seconds = time.perf_counter() - start_time
        policy.usage.check(budget)

        # The current target is not passed to predict and is enqueued only after scoring.
        prediction = policy.predict(sample.features)
        if not math.isfinite(prediction):
            nonfinite_predictions += 1
        squared_error = (prediction - sample.target) ** 2
        trace.append(
            {
                "step": sample.step,
                "task_id": definition.identity.task_id,
                "regime": sample.regime,
                "prediction": prediction,
                "target": sample.target,
                "squared_error": squared_error,
                "label_available_at": sample.label_available_at,
                "updates_before_prediction": policy.usage.parameter_updates,
            }
        )
        pending[sample.label_available_at].append(
            PendingLabel(
                source_step=sample.step,
                available_at=sample.label_available_at,
                features=sample.features,
                target=sample.target,
                prediction=prediction,
            )
        )
        policy.usage.steps += 1

    policy.usage.wall_seconds = time.perf_counter() - start_time
    policy.usage.peak_state_bytes = max(policy.usage.peak_state_bytes, policy.state_bytes())
    policy.usage.check(budget)
    if len(trace) != definition.steps:
        raise RuntimeError("trace length does not match the frozen task length")

    regime_length = definition.regime_length
    switch_window = min(24, regime_length)
    switch_mses = [
        _window_mse(trace, start, start + switch_window)
        for start in (regime_length, 2 * regime_length, 3 * regime_length)
    ]
    initial_late_mse = _window_mse(
        trace,
        regime_length - switch_window,
        regime_length,
    )
    return_mse = _window_mse(
        trace,
        2 * regime_length,
        2 * regime_length + switch_window,
    )
    retention_ratio = return_mse / max(initial_late_mse, 1e-12)
    metrics = {
        "primary": {
            "name": "prequential_mean_squared_error",
            "value": _mean([row["squared_error"] for row in trace]),
            "direction": "lower_is_better",
        },
        "adaptation_switch_mse": _mean(switch_mses),
        "retention_return_mse": return_mse,
        "retention_ratio_to_initial_late": retention_ratio,
        "regime_mse": {
            regime: _mean(
                [row["squared_error"] for row in trace if row["regime"] == regime]
            )
            for regime in ("A_initial", "B_conflict", "A_return", "C_novel")
        },
        "instability": {
            "nonfinite_predictions": nonfinite_predictions,
            "max_absolute_prediction": max(abs(row["prediction"]) for row in trace),
            "max_update_norm": policy.max_update_norm,
        },
        "resource_usage": policy.usage.to_dict(),
        "undelivered_labels_at_horizon": sum(len(labels) for labels in pending.values()),
    }
    return metrics, trace


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "neurogenesis").glob("*.py"))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_development_config(path: Path) -> dict[str, Any]:
    raw = strict_json_load(path)
    if not isinstance(raw, dict):
        raise ContractError("development config must be a JSON object")
    required = {
        "schema_version",
        "split",
        "matrix_id",
        "meta_train_identities",
        "tasks",
        "policies",
    }
    if set(raw) != required:
        raise ContractError("development config fields do not match v1 schema")
    if raw["schema_version"] != "neurogenesis.development.v1":
        raise ContractError("unsupported development config schema")
    if raw["split"] != "development":
        raise ContractError("development entry point accepts only the development split")
    tasks = [TaskDefinition.from_mapping(item) for item in raw["tasks"]]
    meta_identities = [
        TaskDefinition.from_mapping(item).identity for item in raw["meta_train_identities"]
    ]
    validate_identity_sets([meta_identities, [task.identity for task in tasks]])
    if len({task.identity.seed for task in tasks}) < 3:
        raise ContractError("development matrix requires at least three retained seeds")
    if not raw["policies"]:
        raise ContractError("development matrix must contain baselines")
    return raw


def run_development_matrix(
    root: Path,
    protocol_path: Path,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol = Protocol.load(protocol_path)
    config = load_development_config(config_path)
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing attempt bundle: {output_root}"
        )
    output_root.mkdir(parents=True)
    tasks = [TaskDefinition.from_mapping(item) for item in config["tasks"]]
    policies = config["policies"]
    manifest = {
        "schema_version": "neurogenesis.run-bundle.v1",
        "matrix_id": config["matrix_id"],
        "split": config["split"],
        "protocol_sha256": protocol.digest,
        "config_sha256": sha256_json(config),
        "source_sha256": source_digest(root),
        "python": sys.version,
        "platform": platform.platform(),
        "serial_execution": True,
        "attempt_count": len(tasks) * len(policies),
    }
    _write_json(output_root / "run_manifest.json", manifest)

    statuses: list[dict[str, Any]] = []
    for task in tasks:
        for policy_definition in policies:
            policy_name = policy_definition["name"]
            attempt_id = f"{task.identity.task_id}__{policy_name}"
            attempt_dir = output_root / "attempts" / attempt_id
            attempt_dir.mkdir(parents=True)
            attempt = {
                "attempt_id": attempt_id,
                "task": asdict(task),
                "policy": policy_definition,
                "status": "STARTED",
            }
            _write_json(attempt_dir / "attempt.json", attempt)
            try:
                protocol.admit(task.identity.split, policy_definition["kind"])
                policy = build_policy(policy_definition, task.feature_dim)
                metrics, trace = execute_task(task, policy, protocol.run_budget)
                save_checkpoint(policy, attempt_dir / "checkpoint.json")
                _write_jsonl(attempt_dir / "trace.jsonl", trace)
                trace_hash = hashlib.sha256((attempt_dir / "trace.jsonl").read_bytes()).hexdigest()
                result = {
                    "schema_version": "neurogenesis.result.v1",
                    "attempt_id": attempt_id,
                    "status": "SUCCEEDED",
                    "task_id": task.identity.task_id,
                    "seed": task.identity.seed,
                    "family": task.identity.family,
                    "split": task.identity.split,
                    "policy_name": policy_name,
                    "policy_kind": policy_definition["kind"],
                    "metrics": metrics,
                    "trace_sha256": trace_hash,
                    "error": None,
                }
            except Exception as error:  # retained evidence, then continue remaining matrix
                result = {
                    "schema_version": "neurogenesis.result.v1",
                    "attempt_id": attempt_id,
                    "status": "FAILED",
                    "task_id": task.identity.task_id,
                    "seed": task.identity.seed,
                    "family": task.identity.family,
                    "split": task.identity.split,
                    "policy_name": policy_name,
                    "policy_kind": policy_definition.get("kind"),
                    "metrics": None,
                    "trace_sha256": None,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            _write_json(attempt_dir / "result.json", result)
            statuses.append(
                {
                    "attempt_id": attempt_id,
                    "status": result["status"],
                    "result_sha256": hashlib.sha256(
                        (attempt_dir / "result.json").read_bytes()
                    ).hexdigest(),
                }
            )
    receipt = {
        "schema_version": "neurogenesis.run-receipt.v1",
        "matrix_id": config["matrix_id"],
        "attempted": len(statuses),
        "succeeded": sum(row["status"] == "SUCCEEDED" for row in statuses),
        "failed": sum(row["status"] == "FAILED" for row in statuses),
        "statuses": statuses,
    }
    _write_json(output_root / "receipt.json", receipt)
    return receipt

