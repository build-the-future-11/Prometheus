from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INTEGRATION_VERSION = "atlas.integration.v0.1"
RETAINED_VERDICT = "VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION"


@dataclass(frozen=True)
class IntegrationConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def source_ref(self) -> str:
        return self.raw["source_ref"]

    @property
    def forbidden_artifact(self) -> str:
        return self.raw["dataset"]["forbidden_artifact"]

    @property
    def retained_verdict(self) -> str:
        return self.raw["scientific_boundary"]["retained_verdict"]


def _require_nonempty_string(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def load_integration(path: str | Path) -> IntegrationConfig:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid integration JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("integration JSON must contain an object")

    if payload.get("integration_version") != INTEGRATION_VERSION:
        raise ValueError(f"integration_version must equal {INTEGRATION_VERSION!r}")
    _require_nonempty_string(payload, "name", "integration")
    _require_nonempty_string(payload, "repository", "integration")
    source_ref = _require_nonempty_string(payload, "source_ref", "integration")
    if not re.fullmatch(r"[0-9a-f]{40}", source_ref):
        raise ValueError("integration.source_ref must be a full lowercase 40-character Git SHA")
    if payload.get("mode") != "reproduction":
        raise ValueError("integration.mode must be 'reproduction'")

    boundary = payload.get("scientific_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("scientific_boundary must be an object")
    if boundary.get("retained_verdict") != RETAINED_VERDICT:
        raise ValueError(f"retained_verdict must remain {RETAINED_VERDICT}")
    if boundary.get("allow_result_rescue") is not False:
        raise ValueError("allow_result_rescue must remain false")
    if boundary.get("allow_test_split") is not False:
        raise ValueError("allow_test_split must remain false")
    allowed_splits = boundary.get("allowed_splits")
    if not isinstance(allowed_splits, list) or set(allowed_splits) != {"train", "validation"}:
        raise ValueError("allowed_splits must freeze exactly train and validation")
    if boundary.get("new_hypotheses_require_new_protocol") is not True:
        raise ValueError("new_hypotheses_require_new_protocol must remain true")

    environment = payload.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("environment must be an object")
    if _require_nonempty_string(environment, "python", "environment") != "3.11":
        raise ValueError("environment.python must remain '3.11'")
    _require_nonempty_string(environment, "install", "environment")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    _require_nonempty_string(dataset, "manifest", "dataset")
    _require_nonempty_string(dataset, "download_command", "dataset")
    forbidden = _require_nonempty_string(dataset, "forbidden_artifact", "dataset")
    if "test" not in forbidden.casefold():
        raise ValueError("forbidden_artifact must identify the ARC test artifact")

    _require_nonempty_string(payload, "protocol_preflight", "integration")
    producer = payload.get("producer")
    verifier = payload.get("verifier")
    if not isinstance(producer, dict) or not isinstance(verifier, dict):
        raise ValueError("producer and verifier must be objects")
    _require_nonempty_string(producer, "command", "producer")
    _require_nonempty_string(verifier, "command", "verifier")
    seeds = producer.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) and seed >= 0 for seed in seeds):
        raise ValueError("producer.seeds must be a non-empty list of non-negative integers")

    contract = payload.get("executor_contract")
    if not isinstance(contract, list) or not contract or not all(isinstance(item, str) and item.strip() for item in contract):
        raise ValueError("executor_contract must be a non-empty list of strings")
    return IntegrationConfig(path=source, raw=payload)


def _safe_repo_path(repo_root: Path, relative_path: str) -> Path:
    root = repo_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"integration path escapes repository root: {relative_path}") from exc
    return candidate


def assert_forbidden_absent(repo_root: str | Path, config: IntegrationConfig) -> None:
    forbidden = _safe_repo_path(Path(repo_root), config.forbidden_artifact)
    if forbidden.exists():
        raise RuntimeError(f"forbidden ARC test artifact is present: {config.forbidden_artifact}")


def _argument_value(command: str, flag: str) -> str | None:
    parts = shlex.split(command)
    for index, token in enumerate(parts):
        if token == flag and index + 1 < len(parts):
            return parts[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def build_execution_plan(config: IntegrationConfig) -> list[dict[str, str]]:
    return [
        {"name": "install", "command": config.raw["environment"]["install"]},
        {"name": "download", "command": config.raw["dataset"]["download_command"]},
        {"name": "protocol_preflight", "command": config.raw["protocol_preflight"]},
        {"name": "producer", "command": config.raw["producer"]["command"]},
        {"name": "verifier", "command": config.raw["verifier"]["command"]},
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _verify_source(repo_root: Path, config: IntegrationConfig) -> str:
    actual = _git(repo_root, "rev-parse", "HEAD")
    if actual != config.source_ref:
        raise RuntimeError(f"source SHA mismatch: expected {config.source_ref}, found {actual}")
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("source repository has tracked working-tree changes; refusing non-reproducible execution")
    return actual


def _create_run_dir(output_root: Path, source_ref: str, run_id: str | None = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"lam-jepa-arc-v3-{timestamp}-{source_ref[:8]}-{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, '.', '_' and '-'")
    destination = output_root / run_id
    destination.mkdir(parents=False, exist_ok=False)
    return destination


def _copy_workspace(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in {".git", ".venv", "__pycache__", "ci-evidence"}}
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def _venv_bin(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _run_step(
    run_dir: Path,
    workspace: Path,
    env: dict[str, str],
    name: str,
    command: str,
    sequence: int,
) -> dict[str, Any]:
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    stdout_path = logs / f"{sequence:02d}-{name}.stdout.log"
    stderr_path = logs / f"{sequence:02d}-{name}.stderr.log"
    started = datetime.now(timezone.utc).isoformat()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            shlex.split(command),
            cwd=workspace,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    return {
        "name": name,
        "command": command,
        "exit_code": completed.returncode,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stdout": str(stdout_path.relative_to(run_dir)),
        "stderr": str(stderr_path.relative_to(run_dir)),
    }


def _copy_if_present(workspace: Path, relative_path: str | None, destination: Path, logical_name: str) -> dict[str, Any] | None:
    if not relative_path:
        return None
    source = _safe_repo_path(workspace, relative_path)
    if not source.is_file():
        return None
    destination.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix or ".bin"
    target = destination / f"{logical_name}{suffix}"
    shutil.copy2(source, target)
    return {
        "logical_name": logical_name,
        "source_path": relative_path,
        "retained_path": str(target),
    }


def _dataset_hashes(workspace: Path, config: IntegrationConfig) -> list[dict[str, Any]]:
    command = config.raw["producer"]["command"]
    records: list[dict[str, Any]] = []
    for split, flag in (("train", "--train"), ("validation", "--validation")):
        relative = _argument_value(command, flag)
        if relative is None:
            raise RuntimeError(f"producer command does not freeze {flag}")
        path = _safe_repo_path(workspace, relative)
        if not path.is_file():
            raise RuntimeError(f"expected {split} dataset artifact missing after download: {relative}")
        records.append(
            {
                "split": split,
                "path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evidence_manifest(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "evidence_manifest.json" or "workspace" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "atlas.lam-jepa-evidence.v0.1",
        "run_id": run_dir.name,
        "retained_verdict": RETAINED_VERDICT,
        "summary": summary,
        "files": files,
    }


def dry_run_plan(config_path: str | Path) -> dict[str, Any]:
    config = load_integration(config_path)
    return {
        "integration_version": config.raw["integration_version"],
        "repository": config.raw["repository"],
        "source_ref": config.source_ref,
        "retained_verdict": config.retained_verdict,
        "forbidden_artifact": config.forbidden_artifact,
        "steps": build_execution_plan(config),
        "producer_output": _argument_value(config.raw["producer"]["command"], "--out"),
        "verifier_output": _argument_value(config.raw["verifier"]["command"], "--report"),
    }


def execute(
    config_path: str | Path,
    repo_root: str | Path,
    output_root: str | Path,
    *,
    run_id: str | None = None,
) -> Path:
    config = load_integration(config_path)
    repo = Path(repo_root).resolve()
    if not repo.is_dir():
        raise ValueError(f"repository root does not exist: {repo}")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"adapter execution requires Python 3.11, found {sys.version_info.major}.{sys.version_info.minor}")

    source_sha = _verify_source(repo, config)
    assert_forbidden_absent(repo, config)
    run_dir = _create_run_dir(Path(output_root).resolve(), source_sha, run_id)
    workspace = run_dir / "workspace"
    artifacts = run_dir / "artifacts"
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "source_sha": source_sha,
        "repository": config.raw["repository"],
        "retained_verdict": config.retained_verdict,
        "status": "STARTED",
    }

    _write_json(run_dir / "integration.snapshot.json", config.raw)
    _write_json(run_dir / "execution_plan.json", dry_run_plan(config_path))

    try:
        _copy_workspace(repo, workspace)
        assert_forbidden_absent(workspace, config)

        venv_dir = workspace / ".venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        env = os.environ.copy()
        env["PATH"] = str(_venv_bin(venv_dir)) + os.pathsep + env.get("PATH", "")

        sequence = 1
        install = _run_step(run_dir, workspace, env, "install", config.raw["environment"]["install"], sequence)
        records.append(install)
        sequence += 1
        if install["exit_code"] != 0:
            summary["status"] = "INSTALL_FAILED"
            return _finalize(run_dir, workspace, records, summary)

        freeze = _run_step(run_dir, workspace, env, "environment_freeze", "python -m pip freeze", sequence)
        records.append(freeze)
        sequence += 1
        freeze_stdout = run_dir / freeze["stdout"]
        if freeze_stdout.is_file():
            shutil.copy2(freeze_stdout, run_dir / "environment-packages.txt")

        download = _run_step(run_dir, workspace, env, "download", config.raw["dataset"]["download_command"], sequence)
        records.append(download)
        sequence += 1
        assert_forbidden_absent(workspace, config)
        if download["exit_code"] != 0:
            summary["status"] = "DATASET_DOWNLOAD_FAILED"
            return _finalize(run_dir, workspace, records, summary)

        hashes = _dataset_hashes(workspace, config)
        _write_json(run_dir / "dataset_hashes.json", hashes)
        dataset_manifest = config.raw["dataset"]["manifest"]
        _copy_if_present(workspace, dataset_manifest, artifacts, "dataset-manifest")

        preflight = _run_step(run_dir, workspace, env, "protocol_preflight", config.raw["protocol_preflight"], sequence)
        records.append(preflight)
        sequence += 1
        preflight_output = _argument_value(config.raw["protocol_preflight"], "--report")
        _copy_if_present(workspace, preflight_output, artifacts, "protocol-preflight-report")
        assert_forbidden_absent(workspace, config)
        if preflight["exit_code"] != 0:
            summary["status"] = "PROTOCOL_PREFLIGHT_FAILED"
            return _finalize(run_dir, workspace, records, summary)

        producer = _run_step(run_dir, workspace, env, "producer", config.raw["producer"]["command"], sequence)
        records.append(producer)
        sequence += 1
        assert_forbidden_absent(workspace, config)
        producer_output = _argument_value(config.raw["producer"]["command"], "--out")
        producer_record = _copy_if_present(workspace, producer_output, artifacts, "producer-output")

        verifier_record = None
        if producer_record is not None:
            verifier = _run_step(run_dir, workspace, env, "verifier", config.raw["verifier"]["command"], sequence)
            records.append(verifier)
            verifier_output = _argument_value(config.raw["verifier"]["command"], "--report")
            verifier_record = _copy_if_present(workspace, verifier_output, artifacts, "verifier-output")
        assert_forbidden_absent(workspace, config)

        if producer["exit_code"] == 0 and records[-1]["name"] == "verifier" and records[-1]["exit_code"] == 0 and verifier_record:
            summary["status"] = "EXECUTION_COMPLETE"
        elif producer_record is None:
            summary["status"] = "PRODUCER_FAILED_NO_OUTPUT"
        else:
            summary["status"] = "EXECUTION_FAILED_OR_UNVERIFIED"
        return _finalize(run_dir, workspace, records, summary)
    except Exception as exc:
        summary["status"] = "ADAPTER_FAILED"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        _write_json(run_dir / "run_record.json", {"summary": summary, "steps": records})
        if workspace.exists():
            try:
                assert_forbidden_absent(workspace, config)
            except Exception as guard_exc:
                summary["post_run_forbidden_guard"] = str(guard_exc)
        _finalize(run_dir, workspace, records, summary)
        raise


def _finalize(run_dir: Path, workspace: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    _write_json(run_dir / "run_record.json", {"summary": summary, "steps": records})
    if workspace.exists():
        shutil.rmtree(workspace)
    manifest = _evidence_manifest(run_dir, summary)
    _write_json(run_dir / "evidence_manifest.json", manifest)
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atlas frozen LAM-JEPA ARC-v3 execution adapter")
    parser.add_argument("--integration", default="integrations/lam_jepa_arc_v3.json")
    parser.add_argument("--repo", help="Local LAM-JEPA repository root")
    parser.add_argument("--output-root", default="atlas-evidence/lam-jepa-arc-v3")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_plan(args.integration), indent=2, sort_keys=True))
        return 0
    if not args.repo:
        parser.error("--repo is required unless --dry-run is used")
    run_dir = execute(args.integration, args.repo, args.output_root, run_id=args.run_id)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
