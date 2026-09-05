from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "atlas.percy-tasks.v0.1"


def _stable_task_id(run_id: str, experiment_id: str, *, kind: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{kind}\0{experiment_id}".encode("utf-8")).hexdigest()[:16]
    return f"atlas-{kind}-{digest}"


def _require_non_empty_string(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _validate_manifest_dict(manifest: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    run_id = _require_non_empty_string(manifest.get("run_id"), "manifest.run_id")
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("manifest.experiments must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(experiments):
        if not isinstance(raw, Mapping):
            raise ValueError(f"manifest.experiments[{index}] must be an object")
        experiment_id = _require_non_empty_string(raw.get("name"), f"manifest.experiments[{index}].name")
        if experiment_id in seen:
            raise ValueError(f"duplicate experiment name: {experiment_id}")
        seen.add(experiment_id)

        required_artifacts = raw.get("required_artifacts")
        if not isinstance(required_artifacts, (list, tuple)) or not required_artifacts:
            raise ValueError(f"experiment {experiment_id} must declare required_artifacts")
        artifacts = [
            _require_non_empty_string(item, f"{experiment_id}.required_artifacts")
            for item in required_artifacts
        ]

        seed = raw.get("seed")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"experiment {experiment_id} seed must be a non-negative integer")

        normalized.append(
            {
                "name": experiment_id,
                "seed": seed,
                "perturbation": _require_non_empty_string(
                    raw.get("perturbation"), f"{experiment_id}.perturbation"
                ),
                "treatment": _require_non_empty_string(raw.get("treatment"), f"{experiment_id}.treatment"),
                "control": _require_non_empty_string(raw.get("control"), f"{experiment_id}.control"),
                "required_artifacts": artifacts,
            }
        )
    return run_id, normalized


def _render_command(
    template: str,
    *,
    run_id: str,
    experiment: Mapping[str, Any],
    artifact_dir: Path,
) -> str:
    try:
        rendered = template.format(
            run_id=run_id,
            experiment_id=experiment["name"],
            experiment_name=experiment["name"],
            seed=experiment["seed"],
            perturbation=experiment["perturbation"],
            treatment=experiment["treatment"],
            control=experiment["control"],
            artifact_dir=str(artifact_dir),
        )
    except KeyError as exc:
        raise ValueError(f"unknown command-template placeholder: {exc.args[0]}") from exc
    rendered = rendered.strip()
    if not rendered:
        raise ValueError("command template rendered to an empty command")
    return rendered


def build_percy_export(
    manifest: Mapping[str, Any],
    *,
    working_directory: str,
    command_template: str,
    artifact_root: str | Path,
    status_file: str | Path,
    verification_command: str,
    timeout_seconds: int = 3600,
    max_retries: int = 1,
    retry_backoff_seconds: int = 30,
    resources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id, experiments = _validate_manifest_dict(manifest)
    working_directory = _require_non_empty_string(working_directory, "working_directory")
    command_template = _require_non_empty_string(command_template, "command_template")
    verification_command = _require_non_empty_string(verification_command, "verification_command")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be non-negative")

    artifact_root_path = Path(artifact_root)
    status_path = Path(status_file)
    resource_payload = dict(resources or {"class": "heavy", "slots": 1})
    if int(resource_payload.get("slots", 0)) <= 0:
        raise ValueError("resources.slots must be positive")
    _require_non_empty_string(resource_payload.get("class"), "resources.class")

    tasks: list[dict[str, Any]] = []
    for experiment in experiments:
        task_id = _stable_task_id(run_id, experiment["name"], kind="experiment")
        artifact_dir = artifact_root_path / experiment["name"]
        tasks.append(
            {
                "task_id": task_id,
                "kind": "experiment",
                "dependencies": [],
                "working_directory": working_directory,
                "command": _render_command(
                    command_template,
                    run_id=run_id,
                    experiment=experiment,
                    artifact_dir=artifact_dir,
                ),
                "timeout_seconds": timeout_seconds,
                "retry_policy": {
                    "max_retries": max_retries,
                    "backoff_seconds": retry_backoff_seconds,
                },
                "resources": dict(resource_payload),
                "artifact_directory": str(artifact_dir),
                "expected_artifacts": list(experiment["required_artifacts"]),
                "metadata": {
                    "atlas_run_id": run_id,
                    "atlas_experiment_id": experiment["name"],
                    "seed": experiment["seed"],
                    "perturbation": experiment["perturbation"],
                    "treatment": experiment["treatment"],
                    "control": experiment["control"],
                },
            }
        )

    experiment_task_ids = [task["task_id"] for task in tasks]
    verification_task_id = _stable_task_id(run_id, "__verification__", kind="verify")
    tasks.append(
        {
            "task_id": verification_task_id,
            "kind": "verification",
            "dependencies": experiment_task_ids,
            "working_directory": working_directory,
            "command": verification_command,
            "timeout_seconds": 300,
            "retry_policy": {"max_retries": 0, "backoff_seconds": 0},
            "resources": {"class": "light", "slots": 1},
            "artifact_directory": str(artifact_root_path),
            "expected_artifacts": [],
            "verification": {
                "requires_all_dependencies_succeeded": True,
                "required_artifacts_by_task": {
                    task["task_id"]: list(task["expected_artifacts"])
                    for task in tasks
                },
            },
            "metadata": {
                "atlas_run_id": run_id,
                "atlas_experiment_id": None,
                "purpose": "parent verification task",
            },
        }
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "atlas_run_id": run_id,
        "task_count": len(tasks),
        "experiment_task_count": len(experiment_task_ids),
        "verification_task_id": verification_task_id,
        "status_contract": {
            "path": str(status_path),
            "format": "JSON object mapping task_id to terminal status",
            "success_status": "SUCCEEDED",
        },
        "tasks": tasks,
    }


def verify_export_completion(
    export_payload: Mapping[str, Any],
    *,
    task_statuses: Mapping[str, str],
    artifact_root: str | Path,
) -> tuple[bool, list[str]]:
    if export_payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {export_payload.get('schema_version')!r}")

    tasks = export_payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("export payload tasks must be a list")

    errors: list[str] = []
    root = Path(artifact_root)
    for task in tasks:
        if not isinstance(task, Mapping) or task.get("kind") != "experiment":
            continue
        task_id = _require_non_empty_string(task.get("task_id"), "task.task_id")
        status = str(task_statuses.get(task_id, "")).strip().upper()
        if status != "SUCCEEDED":
            errors.append(f"{task_id}: dependency status is {status or 'MISSING'}, expected SUCCEEDED")
            continue

        artifacts = task.get("expected_artifacts")
        if not isinstance(artifacts, list):
            errors.append(f"{task_id}: expected_artifacts is malformed")
            continue

        metadata = task.get("metadata")
        if not isinstance(metadata, Mapping):
            errors.append(f"{task_id}: metadata is malformed")
            continue
        experiment_id = _require_non_empty_string(
            metadata.get("atlas_experiment_id"),
            f"{task_id}.metadata.atlas_experiment_id",
        )
        experiment_root = root / experiment_id
        for relative in artifacts:
            relative_path = Path(_require_non_empty_string(relative, f"{task_id}.expected_artifacts"))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"{task_id}: unsafe artifact path {relative}")
                continue
            candidate = experiment_root / relative_path
            if not candidate.is_file():
                errors.append(f"{task_id}: missing required artifact {relative}")

    return (not errors, errors)


def write_percy_export(payload: Mapping[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
