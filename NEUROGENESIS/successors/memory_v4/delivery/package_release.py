"""Create deterministic core and full v4 reproduction archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = (2026, 10, 21, 0, 0, 0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def include(path: Path, full: bool) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in {"__pycache__", ".pytest_cache"} for part in rel.parts):
        return False
    if path.suffix in {".pyc", ".aux", ".log", ".out"}:
        return False
    if rel.parts[:2] == ("delivery", "release"):
        return False
    if rel.name in {
        "source-v4.zip",
        "neurogenesis-memory-v4-frozen-evidence.ipynb",
    }:
        return False
    if full:
        return True
    if rel.parts and rel.parts[0] == "runs":
        allowed = {
            "admission.json",
            "audit.json",
            "manifest.json",
            "receipt.json",
            "summary.json",
            "verification-exact.json",
            "verification.json",
        }
        return len(rel.parts) == 2 or rel.name in allowed
    if rel.parts[:2] == ("hosted", "kaggle-output"):
        return rel.name in {
            "terminal.json",
            "hosted-verification.json",
            "summary.json",
            "verification-exact.json",
            "audit.json",
            "manifest.json",
            "receipt.json",
        }
    return True


def archive(output: Path, full: bool) -> dict:
    files = [path for path in sorted(ROOT.rglob("*")) if path.is_file() and include(path, full)]
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in files:
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return {"path": output.name, "sha256": digest(output), "bytes": output.stat().st_size, "files": len(files)}


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    core = archive(output / "neurogenesis-memory-v4-core.zip", False)
    full = archive(output / "neurogenesis-memory-v4-full.zip", True)
    manifest = {
        "status": "PACKAGED",
        "root": str(ROOT),
        "core": core,
        "full": full,
        "independent_external_replication": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
