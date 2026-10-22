"""Hash-bound Kaggle launcher for the frozen DLML-S3-R0 GPU study."""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import traceback


RUN_ID = "kaggle-dlml-s3-r0-r1"
WORK = Path("/kaggle/working/derived-learning-s3")
ARCHIVE = Path("/kaggle/working") / f"{RUN_ID}.tar.gz"


def main() -> int:
    bundles = list(Path("/kaggle/input").rglob("dlml-s3-source.bundle"))
    if len(bundles) != 1:
        raise RuntimeError(f"expected one source bundle, found {bundles}")
    bundle = bundles[0]
    expected = bundle.with_name("dlml-s3-source.sha256").read_text().split()[0]
    actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"source hash mismatch: {actual} != {expected}")
    WORK.mkdir(parents=True, exist_ok=False)
    with tarfile.open(bundle) as archive:
        archive.extractall(WORK, filter="data")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(WORK / "src"), PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    status = 1
    try:
        commands = [
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            [sys.executable, "scripts/run_s3_gpu_study.py", "--config", "configs/s3_real_data_gpu.json", "--output-dir", f"runs/{RUN_ID}"],
        ]
        for index, command in enumerate(commands):
            log = WORK / f"host-step-{index}.log"
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=WORK, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=21600)
            print(log.read_text(), flush=True)
            if completed.returncode:
                raise RuntimeError(f"host step {index} failed with exit {completed.returncode}")
        status = 0
    except Exception:
        (WORK / "HOST_FAILURE.txt").write_text(traceback.format_exc())
        raise
    finally:
        with tarfile.open(ARCHIVE, "w:gz") as archive:
            for path in [WORK / "runs", WORK / "HOST_FAILURE.txt", WORK / "host-step-0.log", WORK / "host-step-1.log"]:
                if path.exists():
                    archive.add(path, arcname=path.relative_to(WORK))
        digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
        Path(f"{ARCHIVE}.sha256").write_text(f"{digest}  {ARCHIVE.name}\n")
        print(json.dumps({"run_id": RUN_ID, "status": status, "archive": str(ARCHIVE), "sha256": digest}), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
