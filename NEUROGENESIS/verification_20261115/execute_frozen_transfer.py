"""Reconstruct one frozen v4 transfer and preserve receipts outside frozen code.

This does not edit the source, lock, data, thresholds, or selected methods. Fresh
run ids are mandatory, and a failed command remains a failed command. It is a
local executor. Manuscript outcomes were already exposed, so this is evidence
reconstruction, not fresh protected confirmation, hosted execution, or external
independent reproduction. Frozen producer phase labels are retained as history.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", choices=("air", "beijing"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run-id must be a new plain directory name")
    output = root / "runs" / args.run_id
    if output.exists():
        raise FileExistsError(f"Run already exists: {output}")
    lock_path = root / "runs" / "freeze.json"
    lock = json.loads(lock_path.read_text())
    source = {name: sha(root / name) for name in lock["source"]}
    if source != lock["source"]:
        raise ValueError("Frozen source hash mismatch before execution")
    manifest = json.loads((root / "data" / "manifest.json").read_text())
    dataset = manifest[args.dataset]
    if sha(root / "data" / dataset["file"]) != dataset["sha256"]:
        raise ValueError("Frozen data hash mismatch before execution")
    if json.loads((root / "protocol.json").read_text()) != lock["protocol"]:
        raise ValueError("Frozen protocol differs from lock")
    args.receipt_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "neurogenesis.local-transfer-command-receipt.v1",
        "scope": "Frozen-code local reproduction and evidence reconstruction after manuscript outcomes were exposed; exact replay and separate arithmetic audit",
        "source_repository": "build-the-future-11/Prometheus",
        "source_commit": "3323b449126ab487bc34756eac781b244ddef22b",
        "source_directory": str(root),
        "run_directory": str(output),
        "dataset": args.dataset,
        "new_protected_confirmation": False,
        "historical_outcomes_already_exposed": True,
        "historical_outcome_source": "NEUROGENESIS/successors/memory_v4/paper/main.tex at source_commit",
        "historical_rounded_claims": {"air_relative_mse_change": 0.7430, "beijing_relative_mse_change": -0.1383, "beijing_solve_reduction": 0.6534},
        "frozen_manifest_phase_labels_retained": True,
        "data": dataset,
        "frozen_source_sha256": source,
        "freeze_sha256": sha(lock_path),
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "hosted_execution": False,
        "independent_external_reproduction": False,
        "started_utc": utc(),
        "commands": [],
    }
    save(args.receipt_dir / "admission.json", receipt)
    commands = [
        [sys.executable, "transfer.py", "run", "--dataset", args.dataset,
         "--lock", "runs/freeze.json", "--output", str(output)],
        [sys.executable, "transfer.py", "verify", "--input", str(output),
         "--lock", "runs/freeze.json", "--output", str(output / "verification-exact.json"), "--replay"],
        [sys.executable, "audit.py", str(output), str(output / "audit.json")],
    ]
    for index, command in enumerate(commands, 1):
        started = time.monotonic()
        entry = {"argv": command, "cwd": str(root), "started_utc": utc()}
        log_path = args.receipt_dir / f"command-{index}.log"
        print(json.dumps({"command_index": index, "argv": command, "utc": utc()}), flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        entry.update({"ended_utc": utc(), "wall_seconds": time.monotonic() - started,
                      "returncode": code, "log_sha256": sha(log_path)})
        receipt["commands"].append(entry)
        save(args.receipt_dir / f"command-{index}.json", entry)
        if code:
            break
    receipt["ended_utc"] = utc()
    receipt["all_commands_succeeded"] = len(receipt["commands"]) == 3 and all(
        entry["returncode"] == 0 for entry in receipt["commands"])
    receipt["frozen_source_unchanged"] = all(sha(root / name) == expected
                                             for name, expected in source.items())
    receipt["run_artifact_sha256"] = {
        str(path.relative_to(output)): sha(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    save(args.receipt_dir / "execution.json", receipt)
    print(json.dumps({key: receipt[key] for key in (
        "all_commands_succeeded", "frozen_source_unchanged", "ended_utc")}), flush=True)
    return 0 if receipt["all_commands_succeeded"] and receipt["frozen_source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
