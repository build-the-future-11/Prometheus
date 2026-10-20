"""Fail-closed resource and artifact evidence helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Iterable


class ResourceLimitError(RuntimeError):
    pass


def pageouts() -> int | None:
    proc = Path("/proc/vmstat")
    if proc.exists():
        for line in proc.read_text().splitlines():
            if line.startswith("pswpout "):
                return int(line.split()[1])
    if sys.platform == "darwin":
        completed = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
        for line in completed.stdout.splitlines():
            if line.startswith("Pageouts:"):
                return int(line.split(":", 1)[1].strip().rstrip("."))
    return None


def peak_rss_mib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


class ResourceGuard:
    def __init__(self, max_rss_mib: float, require_no_rising_pageout: bool) -> None:
        self.max_rss_mib = float(max_rss_mib)
        self.require_no_rising_pageout = bool(require_no_rising_pageout)
        self.started_pageouts = pageouts()
        self.started_at = time.time()
        self.observations: list[dict[str, object]] = []

    def check(self, phase: str) -> dict[str, object]:
        rss = peak_rss_mib()
        current_pageouts = pageouts()
        observation = {
            "phase": phase,
            "peak_rss_mib": rss,
            "pageouts": current_pageouts,
            "pageout_delta": None if current_pageouts is None or self.started_pageouts is None else current_pageouts - self.started_pageouts,
            "elapsed_seconds": time.time() - self.started_at,
        }
        self.observations.append(observation)
        if rss > self.max_rss_mib:
            raise ResourceLimitError(f"RSS guard refused {phase}: {rss:.1f} > {self.max_rss_mib:.1f} MiB")
        if self.require_no_rising_pageout:
            if current_pageouts is None or self.started_pageouts is None:
                raise ResourceLimitError(f"pageout counter unavailable at {phase}")
            if current_pageouts > self.started_pageouts:
                raise ResourceLimitError(
                    f"pageout guard refused {phase}: {current_pageouts} > {self.started_pageouts}"
                )
        return observation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def create_manifest(run_dir: Path, excluded: Iterable[str] = ("artifact_manifest.json",)) -> dict[str, object]:
    excluded_set = set(excluded)
    files = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        if relative in excluded_set:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema_version": 1, "files": files}
    write_json(run_dir / "artifact_manifest.json", manifest)
    return manifest


def verify_manifest(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {item["path"]: item for item in manifest["files"]}
    observed = {
        path.relative_to(run_dir).as_posix(): path
        for path in run_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    changed = sorted(
        relative
        for relative, item in expected.items()
        if relative in observed
        and (
            observed[relative].stat().st_size != item["bytes"]
            or sha256_file(observed[relative]) != item["sha256"]
        )
    )
    result = {"valid": not (missing or extra or changed), "missing": missing, "extra": extra, "changed": changed}
    if not result["valid"]:
        raise ValueError(f"artifact verification failed: {result}")
    return result


def environment_record() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pid": os.getpid(),
    }

