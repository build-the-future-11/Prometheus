"""Build a deterministic, secret-free source dataset for hosted execution."""

from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "studies" / "kaggle-source-dataset-v2"
ARCHIVE = OUTPUT / "adaptive-predictive-source-v2.tar.gz"
MANIFEST = OUTPUT / "source-manifest.json"

INCLUDED = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "RESEARCH_STATE.json",
    "docs/DATA_CARD.md",
    "docs/FLAW_AUDIT.md",
    "docs/LITERATURE_REVIEW.md",
    "docs/PROTOCOL_DRAFT.md",
    "docs/PROTOCOL_DRAFT_V0.3.md",
    "protocols/development-v1.json",
    "protocols/development-v2.json",
    "pyproject.toml",
    "uv.lock",
    "src/adaptive_predictive/__init__.py",
    "src/adaptive_predictive/cli.py",
    "src/adaptive_predictive/data.py",
    "src/adaptive_predictive/experiment.py",
    "src/adaptive_predictive/model.py",
    "src/adaptive_predictive/statistics.py",
    "tests/test_data.py",
    "tests/test_experiment.py",
    "tests/test_model.py",
    "tests/test_statistics.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    missing = [name for name in INCLUDED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing source files: {missing}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() or MANIFEST.exists():
        raise SystemExit("refusing to overwrite an existing source dataset bundle")
    files = {name: sha256(ROOT / name) for name in INCLUDED}
    with ARCHIVE.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in INCLUDED:
                    info = archive.gettarinfo(str(ROOT / name), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with (ROOT / name).open("rb") as handle:
                        archive.addfile(info, handle)
    payload = {
        "schema_version": "adaptive-predictive.source-manifest.v1",
        "archive": ARCHIVE.name,
        "archive_sha256": sha256(ARCHIVE),
        "files": files,
        "file_count": len(files),
        "secret_exclusion_policy": "An explicit source allowlist excludes credential files, local data, .git objects, virtual environments, caches, and result directories; external secret scanning is still required before upload.",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

