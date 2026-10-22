"""Kaggle entry point for the frozen ML4-Material Discovery study.

The private input dataset is copied into Kaggle's writable working directory,
then the repository-owned fail-closed runner executes without changing the
scientific protocol. Kaggle persists everything under /kaggle/working as the
notebook output bundle.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


RUN_ID = "kaggle-development-v1-r1"
INPUT_ROOT = Path("/kaggle/input/ml4-material-discovery-source")
WORK_ROOT = Path("/kaggle/working/ml4-material-discovery")


def resolve_input_root() -> Path:
    """Resolve the mounted private dataset by its frozen config signature."""

    if (INPUT_ROOT / "configs" / "study.json").is_file():
        return INPUT_ROOT
    candidates = sorted(Path("/kaggle/input").glob("**/configs/study.json"))
    if len(candidates) == 1:
        return candidates[0].parents[1]
    available = sorted(str(path) for path in Path("/kaggle/input").glob("*"))
    raise FileNotFoundError(
        "unable to resolve the private source dataset; "
        f"expected={INPUT_ROOT}; candidates={candidates}; available={available}"
    )


def main() -> int:
    input_root = resolve_input_root()
    print(f"HOSTED_INPUT_ROOT={input_root}")
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    shutil.copytree(input_root, WORK_ROOT)
    frozen_data = WORK_ROOT / "frozen-data.bundle"
    if not frozen_data.is_file():
        raise FileNotFoundError(f"missing opaque frozen-data bundle: {frozen_data}")
    with tarfile.open(frozen_data, mode="r:") as archive:
        archive.extractall(WORK_ROOT, filter="data")

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{WORK_ROOT / 'src'}:{WORK_ROOT}",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": "/kaggle/working/.matplotlib",
            "XDG_CACHE_HOME": "/kaggle/working/.cache",
            "MPL_IGNORE_SYSTEM_FONTS": "1",
        }
    )
    completed = subprocess.run(
        ["bash", "scripts/run_kaggle.sh", RUN_ID],
        cwd=WORK_ROOT,
        env=environment,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if completed.returncode:
        print(f"HOSTED_RUN_FAILED={RUN_ID}", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
