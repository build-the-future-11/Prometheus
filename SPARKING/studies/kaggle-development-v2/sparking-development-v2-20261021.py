"""Private Kaggle GPU launcher for frozen SPARKING development v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


RUN_ID = "kaggle-development-v2-r1"
WORKING = Path("/kaggle/working")
CHECKOUT = WORKING / "sparking-v2-source"
SOURCE_ROOT = Path("/kaggle/input/sparking-source-v2-20261021")


def run(command: list[str], cwd: Path | None = None, stdout=None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=os.environ.copy(), check=True, stdout=stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_source() -> Path:
    if SOURCE_ROOT.exists():
        return SOURCE_ROOT
    candidates = list(Path("/kaggle/input").rglob("sparking-source-v2.bin"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one SPARKING v2 source archive, found {candidates}")
    return candidates[0].parent


source_root = locate_source()
archive = source_root / "sparking-source-v2.bin"
expected_hash = (source_root / "sparking-source-v2.sha256").read_text().split()[0]
if sha256(archive) != expected_hash:
    raise ValueError("source archive digest mismatch")
if CHECKOUT.exists():
    raise FileExistsError(CHECKOUT)
CHECKOUT.mkdir()
with tarfile.open(archive, "r:gz") as tar:
    members = tar.getmembers()
    if any(member.name.startswith("/") or ".." in Path(member.name).parts for member in members):
        raise ValueError("unsafe archive path")
    tar.extractall(CHECKOUT)

manifest = json.loads((CHECKOUT / "source-manifest.json").read_text())
changed = []
for item in manifest["files"]:
    path = CHECKOUT / item["path"]
    if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        changed.append(item["path"])
if changed:
    raise ValueError(f"source verification failed: {changed}")
expected_files = {item["path"] for item in manifest["files"]} | {"source-manifest.json", "source-revision.txt"}
observed_files = {
    path.relative_to(CHECKOUT).as_posix() for path in CHECKOUT.rglob("*") if path.is_file()
}
extra_files = sorted(observed_files - expected_files)
missing_files = sorted(expected_files - observed_files)
if extra_files or missing_files:
    raise ValueError(f"source inventory drift: extra={extra_files}, missing={missing_files}")
source_verification = {
    "status": "VERIFIED",
    "archive_sha256": expected_hash,
    "git_commit": manifest["git_commit"],
    "file_count": manifest["file_count"],
    "changed": changed,
    "extra": extra_files,
    "missing": missing_files,
}
(WORKING / "source-verification-v2.json").write_text(json.dumps(source_verification, indent=2) + "\n")
os.environ["SPARKING_SOURCE_COMMIT"] = manifest["git_commit"]

run(
    [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-hosted-v2.txt"],
    cwd=CHECKOUT,
)
run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "-e", "."], cwd=CHECKOUT)
with (WORKING / "pip-freeze-v2.txt").open("w") as handle:
    run([sys.executable, "-m", "pip", "freeze", "--all"], cwd=CHECKOUT, stdout=handle)
run([sys.executable, "-m", "pytest", "-q"], cwd=CHECKOUT)
run(
    [
        sys.executable,
        "scripts/run_successor_v2.py",
        "--root",
        str(CHECKOUT),
        "--run-id",
        RUN_ID,
        "--task-type",
        "GPU",
    ],
    cwd=CHECKOUT,
)
run([sys.executable, "scripts/verify_successor_v2.py", str(CHECKOUT / "runs" / RUN_ID)], cwd=CHECKOUT)

result = json.loads((CHECKOUT / "runs" / RUN_ID / "results.json").read_text())
(WORKING / "development-v2-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
bundle = WORKING / f"{RUN_ID}.tar.gz"
with tarfile.open(bundle, "w:gz") as tar:
    tar.add(CHECKOUT / "runs" / RUN_ID, arcname=RUN_ID)
(WORKING / f"{RUN_ID}.sha256").write_text(f"{sha256(bundle)}  {bundle.name}\n")
print(f"SPARKING_V2_COMPLETE={result['scientific_disposition']}", flush=True)
