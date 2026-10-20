"""Hosted development runner; all accepted evidence is emitted to /kaggle/working."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


BRANCH = "codex/ml4materials-research-execution"
REPOSITORY = "https://github.com/build-the-future-11/Atlas.git"
RUN_ID = "kaggle-development-v1-r1"
WORKING = Path("/kaggle/working")
CHECKOUT = WORKING / "atlas-source"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=os.environ.copy())


if CHECKOUT.exists():
    shutil.rmtree(CHECKOUT)
run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)])
project = CHECKOUT / "SPARKING"
os.environ.update(
    {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": "/tmp/sparking-matplotlib",
        "XDG_CACHE_HOME": "/tmp/sparking-cache",
        "LOKY_MAX_CPU_COUNT": str(os.cpu_count() or 2),
        "PYTHONHASHSEED": "0",
    }
)
run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], cwd=project)
run([sys.executable, "-m", "pytest", "-q"], cwd=project)
run([sys.executable, "-m", "sparking", "prepare"], cwd=project)
run(
    [
        sys.executable,
        "-m",
        "sparking",
        "preflight",
        "--output",
        f"artifacts/preflight-{RUN_ID}.json",
    ],
    cwd=project,
)
run([sys.executable, "-m", "sparking", "develop", "--run-id", RUN_ID], cwd=project)
run([sys.executable, "-m", "sparking", "verify", "--run-id", RUN_ID], cwd=project)

environment_path = project / "artifacts" / f"pip-freeze-{RUN_ID}.txt"
environment_path.write_text(
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
)
archive_base = WORKING / RUN_ID
archive = Path(
    shutil.make_archive(
        str(archive_base),
        "gztar",
        root_dir=project,
        base_dir="runs/" + RUN_ID,
    )
)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
(WORKING / f"{RUN_ID}.sha256").write_text(f"{digest}  {archive.name}\n")
summary = json.loads((project / "runs" / RUN_ID / "results.json").read_text())
(WORKING / "development-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(WORKING / f"preflight-{RUN_ID}.json").write_text(
    (project / "artifacts" / f"preflight-{RUN_ID}.json").read_text()
)
print(f"HOSTED_RUN_COMPLETE={RUN_ID}")
print(f"HOSTED_ARCHIVE_SHA256={digest}")

