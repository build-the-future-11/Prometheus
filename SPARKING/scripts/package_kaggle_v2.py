#!/usr/bin/env python3
"""Create the exact private Kaggle source dataset for SPARKING v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


ROOT_FILES = (
    ".gitignore",
    "DATA_CARD.md",
    "INTAKE_AUDIT.md",
    "LICENSE",
    "MODEL_CARD.md",
    "PROTOCOL.md",
    "README.md",
    "RESEARCH_STATE.json",
    "pyproject.toml",
    "requirements-hosted-v2.txt",
    "uv.lock",
)
TREE_PATHS = (
    "src",
    "tests",
)
EXACT_FILES = (
    "data/dataset_receipt.json",
    "data/raw/matbench_expt_gap.json.gz",
    "docs/DEVELOPMENT_DIAGNOSIS_V1.md",
    "docs/DEVELOPMENT_RESULTS_V1.md",
    "protocols/development-v1.json",
    "protocols/development-v2.json",
    "protocols/diagnostic-v1.json",
    "scripts/run_successor_v2.py",
    "scripts/verify_successor_v2.py",
    "artifacts/development-comparison-v1.json",
    "artifacts/diagnostic-verification-v1.json",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def copy_file(root: Path, payload: Path, relative: str) -> None:
    source = root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = payload / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    output.mkdir(parents=True)
    payload = output / "payload"
    payload.mkdir()
    for relative in ROOT_FILES + EXACT_FILES:
        copy_file(root, payload, relative)
    for relative in TREE_PATHS:
        source = root / relative
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
                copy_file(root, payload, path.relative_to(root).as_posix())
    files = []
    for path in sorted(item for item in payload.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(payload).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root.parent, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--", root.name],
        cwd=root.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise RuntimeError(f"SPARKING source must be clean before packaging: {status}")
    manifest = {
        "schema_version": 2,
        "git_commit": commit,
        "file_count": len(files),
        "files": files,
    }
    (payload / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (payload / "source-revision.txt").write_text(commit + "\n")
    archive = output / "sparking-source-v2.bin"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(payload.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(payload).as_posix(), recursive=False)
    archive_hash = digest(archive)
    (output / "sparking-source-v2.sha256").write_text(f"{archive_hash}  {archive.name}\n")
    metadata = {
        "title": "SPARKING source v2 20261021",
        "id": "ryangomez010/sparking-source-v2-20261021",
        "licenses": [{"name": "MIT"}],
        "isPrivate": True,
    }
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.rmtree(payload)
    print(json.dumps({"archive": str(archive), "sha256": archive_hash, "files": len(files), "commit": commit}))


if __name__ == "__main__":
    main()
