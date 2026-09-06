from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .models import RunManifest


@dataclass(frozen=True)
class ArtifactRecord:
    experiment: str
    path: str
    sha256: str
    size_bytes: int


@dataclass
class EvidenceReport:
    run_id: str
    status: str
    expected_artifacts: int
    verified_artifacts: int
    missing: list[str]
    invalid_json: list[str]
    artifacts: list[ArtifactRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "expected_artifacts": self.expected_artifacts,
            "verified_artifacts": self.verified_artifacts,
            "missing": list(self.missing),
            "invalid_json": list(self.invalid_json),
            "artifacts": [asdict(record) for record in self.artifacts],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_experiment_component(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must be a single safe path component")
    windows = PureWindowsPath(value)
    if windows.drive or windows.is_absolute():
        raise ValueError(f"{field} must be a single safe path component")
    return value


def _safe_relative_artifact_path(value: str, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{field} must be a safe relative artifact path")

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    normalized_parts = value.replace("\\", "/").split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in normalized_parts)
    ):
        raise ValueError(f"{field} must be a safe relative artifact path")
    return Path(*normalized_parts)


def verify_evidence(manifest: RunManifest, evidence_root: str | Path) -> EvidenceReport:
    manifest.validate()
    root = Path(evidence_root)
    resolved_root = root.resolve()
    missing: list[str] = []
    invalid_json: list[str] = []
    artifacts: list[ArtifactRecord] = []
    expected = 0

    for experiment in manifest.experiments:
        experiment_name = _safe_experiment_component(experiment.name, "experiment.name")
        experiment_root = root / experiment_name

        artifact_contract: list[tuple[str, Path]] = []
        seen_artifacts: set[str] = set()
        for relative_name in experiment.required_artifacts:
            relative_path = _safe_relative_artifact_path(
                relative_name,
                f"{experiment_name}.required_artifacts",
            )
            normalized = relative_path.as_posix()
            if normalized in seen_artifacts:
                raise ValueError(
                    f"experiment {experiment_name} has duplicate required artifact: {relative_name}"
                )
            seen_artifacts.add(normalized)
            artifact_contract.append((relative_name, relative_path))

        for relative_name, safe_relative_path in artifact_contract:
            expected += 1
            artifact_path = experiment_root / safe_relative_path
            report_path = str(Path(experiment_name) / safe_relative_path)
            if not artifact_path.is_file():
                missing.append(report_path)
                continue

            try:
                resolved_artifact = artifact_path.resolve(strict=True)
                resolved_artifact.relative_to(resolved_root)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"{experiment_name}.required_artifacts escapes evidence root: {relative_name}"
                ) from exc

            if not resolved_artifact.is_file():
                missing.append(report_path)
                continue

            if resolved_artifact.suffix.lower() == ".json":
                try:
                    payload = json.loads(resolved_artifact.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("top-level JSON value must be an object")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    invalid_json.append(report_path)
                    continue

            artifacts.append(
                ArtifactRecord(
                    experiment=experiment_name,
                    path=report_path,
                    sha256=_sha256(resolved_artifact),
                    size_bytes=resolved_artifact.stat().st_size,
                )
            )

    status = "COMPLETE" if not missing and not invalid_json else "INCOMPLETE"
    return EvidenceReport(
        run_id=manifest.run_id,
        status=status,
        expected_artifacts=expected,
        verified_artifacts=len(artifacts),
        missing=sorted(missing),
        invalid_json=sorted(invalid_json),
        artifacts=sorted(artifacts, key=lambda item: (item.experiment, item.path)),
    )
