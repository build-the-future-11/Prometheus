"""Hosted development runner. It never parses protected values dated 2017 or later."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path
import shutil
import subprocess
import sys


RUN_ID = "kaggle-development-v1"
WORKING = Path("/kaggle/working")
CHECKOUT = WORKING / "adaptive-predictive-source"
SOURCE_MOUNT_NAME = "adaptive-predictive-source-v1-20261021"
DATASET_MOUNT_NAME = "hourly-energy-consumption"


def locate_mount(name: str, marker: str) -> Path:
    direct = Path("/kaggle/input") / name
    if direct.exists():
        return direct
    matches = sorted({path.parent for path in Path("/kaggle/input").rglob(marker) if name in str(path)})
    if len(matches) != 1:
        raise RuntimeError(f"expected one mount for {name}, found {matches}")
    return matches[0]


source_mount = locate_mount(SOURCE_MOUNT_NAME, "source-manifest.json")
archive = source_mount / "adaptive-predictive-source.tar.gz"
if CHECKOUT.exists():
    raise FileExistsError(f"refusing to overwrite {CHECKOUT}")
CHECKOUT.mkdir(parents=True)
shutil.unpack_archive(str(archive), str(CHECKOUT), "gztar")
manifest = json.loads((source_mount / "source-manifest.json").read_text())
observed = {
    path.relative_to(CHECKOUT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in CHECKOUT.rglob("*") if path.is_file()
}
if observed != manifest["files"]:
    raise RuntimeError("hosted source does not match the locally verified source manifest")
(WORKING / "source-verification.json").write_text(json.dumps({
    "valid": True,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "file_count": len(observed),
    "files": observed,
}, indent=2, sort_keys=True) + "\n")

sys.path.insert(0, str(CHECKOUT / "src"))
from adaptive_predictive.data import load_pjm_daily  # noqa: E402
from adaptive_predictive.experiment import run_series, write_run  # noqa: E402
from adaptive_predictive.model import DDFSConfig  # noqa: E402
from adaptive_predictive.statistics import paired_block_bootstrap  # noqa: E402

data_mount = locate_mount(DATASET_MOUNT_NAME, "AEP_hourly.csv")
registered = [
    "AEP_hourly.csv", "COMED_hourly.csv", "DAYTON_hourly.csv", "DEOK_hourly.csv",
    "DOM_hourly.csv", "DUQ_hourly.csv", "EKPC_hourly.csv", "FE_hourly.csv",
    "PJME_hourly.csv", "PJMW_hourly.csv",
]
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
write_run(run_dir, results, config, stage="development")
differences = {}
for result in results:
    scale = float(result["mase_scale"])
    differences[str(result["series"])] = [
        (abs(float(row["forecast"]["ddfs"]) - float(row["outcome"]))
         - abs(float(row["forecast"]["fixed_share"]) - float(row["outcome"]))) / scale
        for row in result["predictions"]
    ]
bootstrap = paired_block_bootstrap(
    differences, block_length=7, repetitions=5000, seed=20261021
)
analysis = {
    "run_id": RUN_ID,
    "primary_comparison": "ddfs-minus-fixed_share-mase",
    "bootstrap": asdict(bootstrap),
    "series_differences": {
        name: sum(values) / len(values) for name, values in differences.items()
    },
    "input_manifest": input_manifest,
    "config": asdict(config),
    "claim_status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
}
(WORKING / "development-analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
(WORKING / "environment.txt").write_text(subprocess.run(
    [sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True
).stdout)
archive_path = Path(shutil.make_archive(str(WORKING / RUN_ID), "gztar", root_dir=WORKING, base_dir=RUN_ID))
(WORKING / f"{RUN_ID}.sha256").write_text(
    f"{hashlib.sha256(archive_path.read_bytes()).hexdigest()}  {archive_path.name}\n"
)
print(json.dumps(analysis, indent=2, sort_keys=True))
print(f"HOSTED_RUN_COMPLETE={RUN_ID}")

