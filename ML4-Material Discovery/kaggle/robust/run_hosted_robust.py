"""Execute the source-hash-bound v3 matrix in a fresh Kaggle CPU environment."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile

RUN_ID = "kaggle-robust-aggregation-v3-r1"
WORK = Path("/kaggle/working/ml4-material-discovery-v3")


def main():
    matches = list(Path("/kaggle/input").rglob("robust-source.bundle"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one source bundle, found {len(matches)}")
    source = matches[0]
    expected = source.with_name("robust-source.sha256").read_text().split()[0]
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError("source transport hash mismatch")
    WORK.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source) as archive:
        archive.extractall(WORK, filter="data")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=f"{WORK / 'src'}:{WORK}", MPLBACKEND="Agg",
                       MPLCONFIGDIR="/kaggle/working/.mpl-v3", MPL_IGNORE_SYSTEM_FONTS="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1")
    (WORK / "artifacts").mkdir(exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "joblib", "matplotlib", "pytest"]}
    (WORK / "artifacts/host-environment.json").write_text(json.dumps({"python": sys.version, "versions": versions, "bundle_sha256": expected}, indent=2))
    print("ENVIRONMENT=" + json.dumps(versions), flush=True)
    expected_versions = {"numpy": "2.1.3", "scikit-learn": "1.6.1", "joblib": "1.6.0"}
    if any(versions[p] != v for p, v in expected_versions.items()):
        raise RuntimeError("host dependency versions differ from the frozen execution contract")
    commands = [
        [sys.executable, "-m", "pytest", "-q", "--junitxml=artifacts/pytest-v3.xml"],
        [sys.executable, "scripts/dry_run_robust.py"],
        [sys.executable, "-m", "ml4materials", "run-robust", "--run-id", RUN_ID],
        [sys.executable, "scripts/verify_evidence.py", "--run-id", RUN_ID, "--output", "artifacts/numerical-audit-v3.json"],
    ]
    try:
        for index, command in enumerate(commands):
            log = WORK / "artifacts" / f"host-step-{index}.log"
            with log.open("w") as stream:
                done = subprocess.run(command, cwd=WORK, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
            print(log.read_text(), flush=True)
            if done.returncode:
                raise RuntimeError(f"host step {index} failed with exit {done.returncode}; retained {log}")
    finally:
        archive_path = Path("/kaggle/working") / (RUN_ID + ".tar.gz")
        with tarfile.open(archive_path, "w:gz") as archive:
            for directory in ["runs", "artifacts"]:
                if (WORK / directory).exists():
                    archive.add(WORK / directory, arcname=directory)
        archive_path.with_suffix(".tar.gz.sha256").write_text(hashlib.sha256(archive_path.read_bytes()).hexdigest() + "  " + archive_path.name + "\n")
    print("HOSTED_RUN_COMPLETE=" + RUN_ID, flush=True)


if __name__ == "__main__":
    main()
