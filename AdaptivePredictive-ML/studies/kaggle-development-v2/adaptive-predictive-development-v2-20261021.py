"""Hosted development-v2 runner with mechanism-isolating controls.

The runner never parses protected values dated 2017 or later. It verifies the
source archive before extraction, records every registered series, and evaluates
all preregistered contrasts without selecting a configuration from these outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile


RUN_ID = "kaggle-development-v2"
WORKING = Path("/kaggle/working")
CHECKOUT = WORKING / "adaptive-predictive-source-v2"
SOURCE_MOUNT_NAME = "adaptive-predictive-source-v2-20261021"
DATASET_MOUNT_NAME = "hourly-energy-consumption"


def locate_mount(name: str, marker: str) -> Path:
    direct = Path("/kaggle/input") / name
    if direct.exists():
        return direct
    matches = sorted({path.parent for path in Path("/kaggle/input").rglob(marker) if name in str(path)})
    if len(matches) != 1:
        raise RuntimeError(f"expected one mount for {name}, found {matches}")
    return matches[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_mount = locate_mount(SOURCE_MOUNT_NAME, "source-manifest.json")
archive = source_mount / "adaptive-predictive-source-v2.tar.gz"
manifest = json.loads((source_mount / "source-manifest.json").read_text())
if CHECKOUT.exists():
    raise FileExistsError(f"refusing to overwrite {CHECKOUT}")
if archive.is_file():
    packaging_mode = "retained-archive"
    archive_sha256 = file_sha256(archive)
    if archive_sha256 != manifest["archive_sha256"]:
        raise RuntimeError("hosted source archive digest does not match its manifest")
    CHECKOUT.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as source_archive:
        members = source_archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise RuntimeError(f"unsafe source archive member: {member.name}")
        source_archive.extractall(CHECKOUT)
else:
    packaging_mode = "kaggle-expanded-archive"
    archive_sha256 = None
    expanded = source_mount / "adaptive-predictive-source-v2"
    if not expanded.is_dir():
        candidates = sorted(
            path.parents[2]
            for path in source_mount.rglob("src/adaptive_predictive/model.py")
        )
        if len(candidates) != 1:
            raise RuntimeError(f"expected one expanded source root, found {candidates}")
        expanded = candidates[0]
    shutil.copytree(expanded, CHECKOUT)
observed = {
    path.relative_to(CHECKOUT).as_posix(): file_sha256(path)
    for path in CHECKOUT.rglob("*") if path.is_file()
}
if observed != manifest["files"]:
    raise RuntimeError("hosted source files do not match the locally verified source manifest")
(WORKING / "source-verification.json").write_text(json.dumps({
    "valid": True,
    "packaging_mode": packaging_mode,
    "archive_sha256": archive_sha256,
    "manifest_archive_sha256": manifest["archive_sha256"],
    "file_count": len(observed),
    "files": observed,
}, indent=2, sort_keys=True) + "\n")

test_environment = os.environ.copy()
test_environment["PYTHONPATH"] = str(CHECKOUT / "src")
test_process = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", str(CHECKOUT / "tests")],
    cwd=CHECKOUT,
    env=test_environment,
    capture_output=True,
    text=True,
)
(WORKING / "engineering-tests-v2.json").write_text(json.dumps({
    "command": [sys.executable, "-m", "pytest", "-q", "tests"],
    "returncode": test_process.returncode,
    "stdout": test_process.stdout,
    "stderr": test_process.stderr,
}, indent=2, sort_keys=True) + "\n")
if test_process.returncode != 0:
    raise RuntimeError("hosted engineering tests failed; see engineering-tests-v2.json")

sys.path.insert(0, str(CHECKOUT / "src"))
from adaptive_predictive.data import load_pjm_daily  # noqa: E402
from adaptive_predictive.experiment import run_series, write_run  # noqa: E402
from adaptive_predictive.model import DDFSConfig  # noqa: E402
from adaptive_predictive.statistics import paired_block_bootstrap  # noqa: E402

registered = [
    "AEP_hourly.csv", "COMED_hourly.csv", "DAYTON_hourly.csv", "DEOK_hourly.csv",
    "DOM_hourly.csv", "DUQ_hourly.csv", "EKPC_hourly.csv", "FE_hourly.csv",
    "PJME_hourly.csv", "PJMW_hourly.csv",
]
data_mount = locate_mount(DATASET_MOUNT_NAME, "AEP_hourly.csv")
config = DDFSConfig()
results = []
input_manifest = {}
for name in registered:
    path = data_mount / name
    if not path.is_file():
        candidates = list(data_mount.rglob(name))
        if len(candidates) != 1:
            raise FileNotFoundError(f"could not resolve {name}: {candidates}")
        path = candidates[0]
    spec = load_pjm_daily(path, start=date(2015, 1, 1), end=date(2016, 12, 31))
    assert spec.timestamps is not None
    evaluate_from = spec.timestamps.index("2016-01-01")
    result = run_series(spec, config, evaluate_from=evaluate_from)
    results.append(result)
    input_manifest[name] = {
        "sha256": spec.source_sha256,
        "n_daily": len(spec.values),
        "first": spec.timestamps[0],
        "last": spec.timestamps[-1],
    }

run_dir = WORKING / RUN_ID
write_run(run_dir, results, config, stage="development-v2")
contrast_pairs = {
    "ddfs_minus_fixed_share": ("ddfs", "fixed_share"),
    "ddfs_minus_adaptive_share": ("ddfs", "adaptive_share"),
    "ddfs_minus_directed_fixed_share": ("ddfs", "directed_fixed_share"),
    "directed_fixed_share_minus_fixed_share": ("directed_fixed_share", "fixed_share"),
    "adaptive_share_minus_fixed_share": ("adaptive_share", "fixed_share"),
    "ddfs_minus_adahedge": ("ddfs", "adahedge"),
}
contrasts = {}
for contrast_name, (left, right) in contrast_pairs.items():
    differences = {}
    for result in results:
        scale = float(result["mase_scale"])
        differences[str(result["series"])] = [
            (
                abs(float(row["forecast"][left]) - float(row["outcome"]))
                - abs(float(row["forecast"][right]) - float(row["outcome"]))
            ) / scale
            for row in result["predictions"]
        ]
    bootstrap = paired_block_bootstrap(
        differences,
        block_length=7,
        repetitions=5000,
        seed=20261021,
        synchronized=True,
    )
    series_differences = {
        name: sum(values) / len(values) for name, values in differences.items()
    }
    contrasts[contrast_name] = {
        "bootstrap": asdict(bootstrap),
        "series_differences": series_differences,
        "series_wins": sum(value < 0.0 for value in series_differences.values()),
        "series_count": len(series_differences),
    }

mean_mase = {
    method: sum(float(result["metrics"][method]["mase"]) for result in results) / len(results)
    for method in results[0]["metrics"]
}
primary = contrasts["ddfs_minus_fixed_share"]["bootstrap"]
relative_improvement = (mean_mase["fixed_share"] - mean_mase["ddfs"]) / mean_mase["fixed_share"]
directed = contrasts["ddfs_minus_adaptive_share"]
analysis = {
    "run_id": RUN_ID,
    "protocol_id": "pjm-daily-development-v2",
    "stage": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
    "contrasts": contrasts,
    "mean_series_weighted_mase": mean_mase,
    "registered_primary_threshold_met": (
        float(primary["estimate"]) <= -0.01 and relative_improvement >= 0.01
    ),
    "directed_mechanism_development_gate_met": (
        float(directed["bootstrap"]["estimate"]) < 0.0
        and int(directed["series_wins"]) * 2 >= int(directed["series_count"])
    ),
    "relative_improvement_vs_fixed_share": relative_improvement,
    "input_manifest": input_manifest,
    "config": asdict(config),
    "claim_status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
}
(WORKING / "development-analysis-v2.json").write_text(
    json.dumps(analysis, indent=2, sort_keys=True) + "\n"
)
(WORKING / "environment-v2.txt").write_text(subprocess.run(
    [sys.executable, "-m", "pip", "freeze", "--all"],
    capture_output=True,
    text=True,
    check=True,
).stdout)
archive_path = Path(shutil.make_archive(str(WORKING / RUN_ID), "gztar", root_dir=WORKING, base_dir=RUN_ID))
(WORKING / f"{RUN_ID}.sha256").write_text(
    f"{file_sha256(archive_path)}  {archive_path.name}\n"
)
print(json.dumps(analysis, indent=2, sort_keys=True))
print(f"HOSTED_RUN_COMPLETE={RUN_ID}")
