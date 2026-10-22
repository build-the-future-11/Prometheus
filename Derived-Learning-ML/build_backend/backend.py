"""Minimal deterministic PEP 517 backend for this dependency-free package."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import tarfile
import zipfile
from pathlib import Path


NAME = "derived-learning-ml"
VERSION = "0.1.0"
DIST_INFO = "derived_learning_ml-0.1.0.dist-info"
WHEEL_NAME = "derived_learning_ml-0.1.0-py3-none-any.whl"
ROOT = Path(__file__).resolve().parents[1]


def _metadata() -> bytes:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return (
        "Metadata-Version: 2.3\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: Dependency-free, reproducible experiments for analytically derived learning targets\n"
        "Requires-Python: >=3.11\n"
        "License: MIT\n"
        "Description-Content-Type: text/markdown\n"
        "\n"
        f"{readme}"
    ).encode("utf-8")


def _wheel_metadata() -> bytes:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: derived-learning-minimal-backend 1\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode("utf-8")


def _entry_points() -> bytes:
    return b"[console_scripts]\nderived-learning = derived_learning.cli:main\n"


def _hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _wheel_files() -> dict[str, bytes]:
    files = {
        str(path.relative_to(ROOT / "src")): path.read_bytes()
        for path in sorted((ROOT / "src" / "derived_learning").glob("*.py"))
    }
    files[f"{DIST_INFO}/METADATA"] = _metadata()
    files[f"{DIST_INFO}/WHEEL"] = _wheel_metadata()
    files[f"{DIST_INFO}/entry_points.txt"] = _entry_points()
    return files


def _record(files: dict[str, bytes]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, data in sorted(files.items()):
        writer.writerow([name, _hash(data), len(data)])
    writer.writerow([f"{DIST_INFO}/RECORD", "", ""])
    return stream.getvalue().encode("utf-8")


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings=None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None) -> str:
    destination = Path(metadata_directory) / DIST_INFO
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "METADATA").write_bytes(_metadata())
    (destination / "WHEEL").write_bytes(_wheel_metadata())
    (destination / "entry_points.txt").write_bytes(_entry_points())
    return DIST_INFO


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    destination = Path(wheel_directory) / WHEEL_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = _wheel_files()
    files[f"{DIST_INFO}/RECORD"] = _record(files)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return WHEEL_NAME


def build_sdist(sdist_directory, config_settings=None) -> str:
    filename = "derived_learning_ml-0.1.0.tar.gz"
    destination = Path(sdist_directory) / filename
    included = [
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        *sorted((ROOT / "src" / "derived_learning").glob("*.py")),
        ROOT / "build_backend" / "backend.py",
    ]
    with destination.open("wb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            for path in included:
                data = path.read_bytes()
                relative = Path("derived_learning_ml-0.1.0") / path.relative_to(ROOT)
                info = tarfile.TarInfo(str(relative))
                info.size = len(data)
                info.mode = 0o644
                info.mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
                archive.addfile(info, io.BytesIO(data))
    return filename

