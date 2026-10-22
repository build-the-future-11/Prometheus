#!/usr/bin/env python3
"""Verify immutable member hashes in an assembled NEUROGENESIS submission package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(package: Path) -> dict:
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        return {"schema_version": "neurogenesis.submission-verification.v1", "status": "INVALID", "errors": ["manifest missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {})
    observed = {
        path.relative_to(package).as_posix(): _hash(path)
        for path in package.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    errors = []
    for name in sorted(set(expected) | set(observed)):
        if expected.get(name) != observed.get(name):
            errors.append(f"package member mismatch: {name}")
    readiness = package / "SUBMISSION_READINESS.md"
    if not readiness.is_file() or "NOT SUBMISSION READY" not in readiness.read_text(encoding="utf-8"):
        errors.append("explicit non-readiness decision missing")
    return {
        "schema_version": "neurogenesis.submission-verification.v1",
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "member_count": len(observed),
        "analysis_artifact_map_sha256": manifest.get("analysis_artifact_map_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verification receipt: {args.output}")
    result = verify(args.package.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
