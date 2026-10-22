"""Safely extract and independently verify the downloaded Kaggle evidence archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import read, sha, verify  # noqa: E402
from transfer import verify as verify_transfer  # noqa: E402


def fail(message: str) -> None:
    raise ValueError(message)


def safe_extract(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    target = output.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            destination = (target / item.filename).resolve()
            if destination != target and target not in destination.parents:
                fail(f"unsafe archive path: {item.filename}")
        zipped.extractall(target)


def check_receipts(run: Path, check: dict) -> dict:
    exact = read(run / "verification-exact.json")
    audit = read(run / "audit.json")
    if (
        check.get("status") != "VALID"
        or check.get("failed")
        or exact.get("status") != "VALID"
        or exact.get("failed")
        or not exact.get("exact_replay")
        or exact.get("artifact_map_sha256") != check.get("artifact_map_sha256")
        or audit.get("status") != "VALID"
        or audit.get("failed")
    ):
        fail(f"verification or audit gate failed: {run}")
    return {"verification": exact, "audit": audit}


def main(archive: Path, output: Path, receipt: Path) -> None:
    archive = archive.resolve()
    safe_extract(archive, output)
    terminal = read(output / "terminal.json")
    if terminal.get("status") != "EVIDENCE_RETAINED" or terminal.get("host") != "kaggle":
        fail("hosted terminal status is not complete Kaggle evidence")

    delivery = read(ROOT / "delivery" / "manifest.json")
    embedded = output / "source-v4.zip"
    if sha(embedded) != delivery["source_archive_sha256"]:
        fail("embedded source archive does not match uploaded archive identity")

    lock = output / "runs" / "freeze.json"
    frozen = read(lock)
    for relative, expected in frozen["source"].items():
        hosted_file = output / relative
        local_file = ROOT / relative
        if sha(hosted_file) != expected or sha(local_file) != expected:
            fail(f"frozen source identity mismatch: {relative}")
    data_manifest = read(output / "data" / "manifest.json")
    for entry in data_manifest.values():
        relative = Path("data") / entry["file"]
        expected = entry["sha256"]
        hosted_file = output / relative
        local_file = ROOT / relative
        if sha(hosted_file) != expected or sha(local_file) != expected:
            fail(f"frozen data identity mismatch: {relative}")

    confirmation = output / "runs" / "confirmation"
    air = output / "runs" / "air-regression"
    beijing = output / "runs" / "beijing-transfer"
    checks = {
        "confirmation": check_receipts(confirmation, verify(confirmation)),
        "air": check_receipts(air, verify_transfer(air, lock)),
        "beijing": check_receipts(beijing, verify_transfer(beijing, lock)),
    }

    payload = {
        "status": "VALID",
        "platform": "kaggle",
        "archive": str(archive),
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "source_archive_sha256": sha(embedded),
        "exact_replay_valid": True,
        "audits_valid": True,
        "artifact_maps": {
            key: value["verification"]["artifact_map_sha256"] for key, value in checks.items()
        },
        "terminal_sha256": sha(output / "terminal.json"),
        "independent_external_replication": False,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as file:
        json.dump(payload, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.archive, arguments.output, arguments.receipt)
