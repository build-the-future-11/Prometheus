"""Kaggle entry point for the separately frozen direct-quantile successor."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


RUN_ID = "kaggle-quantile-composition-v2-r1"
EXPECTED_INPUT = Path("/kaggle/input/ml4-material-discovery-source")
WORK_ROOT = Path("/kaggle/working/ml4-material-discovery-quantile")


def resolve_input_root() -> Path:
    if (EXPECTED_INPUT / "configs" / "studies" / "quantile-composition-v2.json").is_file():
        return EXPECTED_INPUT
    candidates = sorted(
        Path("/kaggle/input").glob("**/configs/studies/quantile-composition-v2.json")
    )
    if len(candidates) == 1:
        return candidates[0].parents[2]
    available = sorted(str(path) for path in Path("/kaggle/input").glob("*"))
    raise FileNotFoundError(
        "unable to resolve the private source dataset; "
        f"expected={EXPECTED_INPUT}; candidates={candidates}; available={available}"
    )


def run(arguments: list[str], environment: dict[str, str]) -> None:
    subprocess.run(
        arguments,
        cwd=WORK_ROOT,
        env=environment,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
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
            "MPLCONFIGDIR": "/kaggle/working/.matplotlib-quantile",
            "XDG_CACHE_HOME": "/kaggle/working/.cache-quantile",
            "MPL_IGNORE_SYSTEM_FONTS": "1",
        }
    )
    python = sys.executable
    config = "configs/studies/quantile-composition-v2.json"
    preflight = f"artifacts/preflight-{RUN_ID}.json"
    registry = f"artifacts/run-registry-{RUN_ID}.json"
    run([python, "-m", "pytest", "-q"], environment)
    run([python, "-m", "ml4materials", "prepare"], environment)
    run(
        [python, "-m", "ml4materials", "preflight", "--config", config, "--output", preflight],
        environment,
    )
    run(
        [
            python,
            "-m",
            "ml4materials",
            "run-quantile",
            "--run-id",
            RUN_ID,
            "--config",
            config,
        ],
        environment,
    )
    run([python, "-m", "ml4materials", "verify", "--run-id", RUN_ID], environment)
    run([python, "-m", "ml4materials", "registry", "--output", registry], environment)

    archive_path = WORK_ROOT / "artifacts" / f"{RUN_ID}.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        archive.add(WORK_ROOT / "runs" / RUN_ID, arcname=f"runs/{RUN_ID}")
        archive.add(WORK_ROOT / preflight, arcname=preflight)
        archive.add(WORK_ROOT / registry, arcname=registry)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{digest}  artifacts/{archive_path.name}\n", encoding="utf-8"
    )
    print(f"HOSTED_RUN_COMPLETE={RUN_ID}")
    print(f"HOSTED_ARCHIVE={archive_path.relative_to(WORK_ROOT)}")
    print(f"HOSTED_ARCHIVE_SHA256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
