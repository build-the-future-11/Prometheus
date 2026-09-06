from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

SCHEMA_VERSION = "atlas.percy-tasks.v0.1"


def _stable_task_id(run_id: str, experiment_id: str, *, kind: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{kind}\0{experiment_id}".encode("utf-8")).hexdigest()[:16]
    return f"atlas-{kind}-{digest}"


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a non-empty string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _require_safe_component(value: Any, field: str) -> str:
    text = _require_non_empty_string(value, field)
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{field} must be a single safe path component")
    windows = PureWindowsPath(text)
    if windows.drive or windows.is_absolute():
        raise ValueError(f"{field} must be a single safe path component")
    return text


def _safe_relative_artifact_path(value: Any, field: str) -> Path:
    text = _require_non_empty_string(value, field)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    normalized_parts = text.replace("\\", "/").split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in normalized_parts)
    ):
        raise ValueError(f"{field} must be a safe relative artifact path")
    return Path(*normalized_parts)


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
        experiment_id = _require_safe_component(
            raw.get("name"), f"manifest.experiments[{index}].name"
        )
        if experiment_id in seen:
            raise ValueError(f"duplicate experiment name: {experiment_id}")
        seen.add(experiment_id)

        required_artifacts = raw.get("required_artifacts")
        if not isinstance(required_artifacts, (list, tuple)) or not required_artifacts:
            raise ValueError(f"experiment {experiment_id} must declare required_artifacts")
        artifacts: list[str] = []
        seen_artifacts: set[str] = set()
        for item in required_artifacts:
            artifact_text = _require_non_empty_string(item, f"{experiment_id}.required_artifacts")
            _safe_relative_artifact_path(artifact_text, f"{experiment_id}.required_artifacts")
            if artifact_text in seen_artifacts:
                raise ValueError(f"experiment {experiment_id} has duplicate required artifact: {artifact_text}")
            seen_artifacts.add(artifact_text)
            artifacts.append(artifact_text)

        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
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
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if (
        isinstance(retry_backoff_seconds, bool)
        or not isinstance(retry_backoff_seconds, int)
        or retry_backoff_seconds < 0
    ):
        raise ValueError("retry_backoff_seconds must be a non-negative integer")

    artifact_root_path = Path(artifact_root)
    status_path = Path(status_file)
    if resources is None:
        resource_payload = {"class": "heavy", "slots": 1}
    elif not isinstance(resources, Mapping):
        raise ValueError("resources must be an object")
    else:
        resource_payload = dict(resources)
    slots = resource_payload.get("slots")
    if isinstance(slots, bool) or not isinstance(slots, int) or slots <= 0:
        raise ValueError("resources.slots must be a positive integer")
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


def _validated_experiment_tasks(export_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if export_payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {export_payload.get('schema_version')!r}")
    run_id = _require_non_empty_string(export_payload.get("atlas_run_id"), "export.atlas_run_id")

    tasks = export_payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("export payload tasks must be a non-empty list")
    if export_payload.get("task_count") != len(tasks):
        raise ValueError("export payload task_count does not match tasks")

    task_ids: set[str] = set()
    experiments: list[Mapping[str, Any]] = []
    verifiers: list[Mapping[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ValueError(f"export payload tasks[{index}] must be an object")
        task_id = _require_non_empty_string(task.get("task_id"), f"tasks[{index}].task_id")
        if task_id in task_ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        task_ids.add(task_id)
        kind = task.get("kind")
        if kind == "experiment":
            experiments.append(task)
        elif kind == "verification":
            verifiers.append(task)
        else:
            raise ValueError(f"{task_id}: unsupported task kind {kind!r}")

    if not experiments:
        raise ValueError("export payload must contain experiment tasks")
    if export_payload.get("experiment_task_count") != len(experiments):
        raise ValueError("export payload experiment_task_count does not match experiment tasks")
    if len(verifiers) != 1:
        raise ValueError("export payload must contain exactly one verification task")

    verifier = verifiers[0]
    verifier_id = _require_non_empty_string(verifier.get("task_id"), "verification.task_id")
    expected_verifier_id = _stable_task_id(run_id, "__verification__", kind="verify")
    if verifier_id != expected_verifier_id:
        raise ValueError("verification task_id does not match deterministic stable identity")
    if export_payload.get("verification_task_id") != verifier_id:
        raise ValueError("export payload verification_task_id does not match verification task")
    verifier_metadata = verifier.get("metadata")
    if not isinstance(verifier_metadata, Mapping):
        raise ValueError("verification metadata is malformed")
    verifier_run_id = _require_non_empty_string(
        verifier_metadata.get("atlas_run_id"), "verification.metadata.atlas_run_id"
    )
    if verifier_run_id != run_id:
        raise ValueError("verification metadata.atlas_run_id does not match export atlas_run_id")

    expected_dependencies = {_require_non_empty_string(task.get("task_id"), "experiment.task_id") for task in experiments}
    dependencies = verifier.get("dependencies")
    if not isinstance(dependencies, list) or set(dependencies) != expected_dependencies or len(dependencies) != len(expected_dependencies):
        raise ValueError("verification task dependencies do not exactly match experiment tasks")

    verification = verifier.get("verification")
    if not isinstance(verification, Mapping) or verification.get("requires_all_dependencies_succeeded") is not True:
        raise ValueError("verification task must require all dependencies to succeed")
    required_by_task = verification.get("required_artifacts_by_task")
    if not isinstance(required_by_task, Mapping) or set(required_by_task) != expected_dependencies:
        raise ValueError("verification artifact contract does not exactly match experiment tasks")

    seen_experiments: set[str] = set()
    for task in experiments:
        task_id = _require_non_empty_string(task.get("task_id"), "experiment.task_id")
        metadata = task.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{task_id}: metadata is malformed")
        task_run_id = _require_non_empty_string(
            metadata.get("atlas_run_id"), f"{task_id}.metadata.atlas_run_id"
        )
        if task_run_id != run_id:
            raise ValueError(f"{task_id}: metadata.atlas_run_id does not match export atlas_run_id")
        experiment_id = _require_safe_component(
            metadata.get("atlas_experiment_id"),
            f"{task_id}.metadata.atlas_experiment_id",
        )
        if experiment_id in seen_experiments:
            raise ValueError(f"duplicate atlas_experiment_id: {experiment_id}")
        seen_experiments.add(experiment_id)
        expected_task_id = _stable_task_id(run_id, experiment_id, kind="experiment")
        if task_id != expected_task_id:
            raise ValueError(f"{task_id}: task_id does not match deterministic stable identity")

        artifacts = task.get("expected_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"{task_id}: expected_artifacts must be a non-empty list")
        normalized_artifacts = [
            str(_safe_relative_artifact_path(item, f"{task_id}.expected_artifacts"))
            for item in artifacts
        ]
        if len(set(normalized_artifacts)) != len(normalized_artifacts):
            raise ValueError(f"{task_id}: expected_artifacts contains duplicates")

        declared = required_by_task.get(task_id)
        if not isinstance(declared, list) or declared != artifacts:
            raise ValueError(f"{task_id}: verification artifact contract drifted from expected_artifacts")

    return experiments


def verify_export_completion(
    export_payload: Mapping[str, Any],
    *,
    task_statuses: Mapping[str, str],
    artifact_root: str | Path,
) -> tuple[bool, list[str]]:
    experiments = _validated_experiment_tasks(export_payload)

    errors: list[str] = []
    root = Path(artifact_root)
    try:
        resolved_root = root.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return False, [f"artifact root does not exist or cannot be resolved: {root}"]

    for task in experiments:
        task_id = _require_non_empty_string(task.get("task_id"), "task.task_id")
        status = str(task_statuses.get(task_id, "")).strip().upper()
        if status != "SUCCEEDED":
            errors.append(f"{task_id}: dependency status is {status or 'MISSING'}, expected SUCCEEDED")
            continue

        artifacts = task["expected_artifacts"]
        metadata = task["metadata"]
        experiment_id = _require_safe_component(
            metadata.get("atlas_experiment_id"),
            f"{task_id}.metadata.atlas_experiment_id",
        )
        experiment_root = root / experiment_id

        for relative in artifacts:
            try:
                relative_path = _safe_relative_artifact_path(relative, f"{task_id}.expected_artifacts")
            except ValueError:
                errors.append(f"{task_id}: unsafe artifact path {relative}")
                continue
            candidate = experiment_root / relative_path
            try:
                resolved_candidate = candidate.resolve(strict=True)
            except (FileNotFoundError, OSError):
                errors.append(f"{task_id}: missing required artifact {relative}")
                continue
            try:
                resolved_candidate.relative_to(resolved_root)
            except ValueError:
                errors.append(f"{task_id}: artifact escapes artifact root {relative}")
                continue
            # Keep the owner boundary lexical beneath the resolved run root:
            # resolving it would let a sibling-directory symlink redefine ownership.
            try:
                resolved_candidate.relative_to(resolved_root / experiment_id)
            except ValueError:
                errors.append(f"{task_id}: artifact escapes experiment root {relative}")
                continue
            if not resolved_candidate.is_file():
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