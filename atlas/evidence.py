from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
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


def verify_evidence(manifest: RunManifest, evidence_root: str | Path) -> EvidenceReport:
    manifest.validate()
    root = Path(evidence_root)
    missing: list[str] = []
    invalid_json: list[str] = []
    artifacts: list[ArtifactRecord] = []
    expected = 0

    for experiment in manifest.experiments:
        experiment_root = root / experiment.name
        for relative_name in experiment.required_artifacts:
            expected += 1
            artifact_path = experiment_root / relative_name
            relative_path = str(artifact_path.relative_to(root))
            if not artifact_path.is_file():
                missing.append(relative_path)
                continue

            if artifact_path.suffix.lower() == ".json":
                try:
                    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("top-level JSON value must be an object")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    invalid_json.append(relative_path)
                    continue

            artifacts.append(
                ArtifactRecord(
                    experiment=experiment.name,
                    path=relative_path,
                    sha256=_sha256(artifact_path),
                    size_bytes=artifact_path.stat().st_size,
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
