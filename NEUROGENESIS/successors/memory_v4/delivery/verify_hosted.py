"""Safely extract and independently verify the downloaded Kaggle evidence archive.

The transfer adapter adds a fresh raw-artifact binding after exact replay. It
preserves all frozen scientific modules and historical verification receipts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import read, sha, verify  # noqa: E402
from transfer import verify as verify_transfer  # noqa: E402
from receipt_contract import check_receipts, verify_transfer_bound  # noqa: E402


def fail(message: str) -> None:
    raise ValueError(message)


def safe_extract(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    target = output.resolve()
    with zipfile.ZipFile(archive) as zipped:
        seen = set()
        for item in zipped.infolist():
            destination = (target / item.filename).resolve()
            if destination != target and target not in destination.parents:
                fail(f"unsafe archive path: {item.filename}")
            if item.filename in seen:
                fail(f"duplicate archive member: {item.filename}")
            seen.add(item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                fail(f"symlink archive member: {item.filename}")
        zipped.extractall(target)


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
        if sha(output / relative) != expected or sha(ROOT / relative) != expected:
            fail(f"frozen data identity mismatch: {relative}")
    confirmation = output / "runs" / "confirmation"
    air = output / "runs" / "air-regression"
    beijing = output / "runs" / "beijing-transfer"
    checks = {
        "confirmation": check_receipts(confirmation, verify(confirmation)),
        "air": check_receipts(air, verify_transfer_bound(air, lock, verify_transfer)),
        "beijing": check_receipts(beijing, verify_transfer_bound(beijing, lock, verify_transfer)),
    }
    payload = {
        "status": "VALID", "platform": "kaggle", "archive": str(archive),
        "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
        "source_archive_sha256": sha(embedded), "exact_replay_valid": True,
        "audits_valid": True,
        "artifact_maps": {key: value["verification"]["artifact_map_sha256"] for key, value in checks.items()},
        "transfer_binding_schema": "prometheus.transfer-binding.v1",
        "transfer_binding_origin": "fresh local exact replay; historical receipts unchanged",
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
