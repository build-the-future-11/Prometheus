#!/usr/bin/env python3
"""Independent stdlib verifier for a retained NEUROGENESIS development bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "neurogenesis").glob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path, bundle: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest = load(bundle / "run_manifest.json")
    receipt = load(bundle / "receipt.json")
    protocol = load(root / "configs" / "protocol.json")
    config = load(root / "configs" / "development.json")

    if manifest["protocol_sha256"] != json_hash(protocol):
        errors.append("protocol hash mismatch")
    if manifest["config_sha256"] != json_hash(config):
        errors.append("development config hash mismatch")
    if manifest["source_sha256"] != source_hash(root):
        errors.append("producer source hash mismatch")
    if not manifest.get("serial_execution"):
        errors.append("manifest does not bind serial execution")
    if protocol["status"] != "SCOPE_UNRESOLVED" or protocol["approved_mechanism"] is not None:
        errors.append("scope resolution changed relative to this bundle")
    if protocol["protected_evaluation"]["run_count"] != 0:
        errors.append("protected evaluation run count is not zero")

    status_rows = receipt["statuses"]
    if len(status_rows) != manifest["attempt_count"] or receipt["attempted"] != len(status_rows):
        errors.append("attempt count mismatch")
    if len({row["attempt_id"] for row in status_rows}) != len(status_rows):
        errors.append("duplicate attempt identity")

    verified_attempts = 0
    verified_trace_rows = 0
    admitted_kinds = {
        "no_adaptation",
        "standard_sgd",
        "mechanism_off_standard_adam",
    }
    max_state_bytes = protocol["run_budget"]["max_state_bytes"]
    max_steps = protocol["run_budget"]["max_steps"]
    for status_row in status_rows:
        attempt_dir = bundle / "attempts" / status_row["attempt_id"]
        result_path = attempt_dir / "result.json"
        if file_hash(result_path) != status_row["result_sha256"]:
            errors.append(f"result hash mismatch: {status_row['attempt_id']}")
            continue
        result = load(result_path)
        if result["status"] != status_row["status"]:
            errors.append(f"status mismatch: {status_row['attempt_id']}")
        if result["split"] != "development":
            errors.append(f"non-development result: {status_row['attempt_id']}")
        if result["policy_kind"] not in admitted_kinds:
            errors.append(f"unadmitted policy: {status_row['attempt_id']}")
        if result["status"] != "SUCCEEDED":
            verified_attempts += 1
            continue
        trace_path = attempt_dir / "trace.jsonl"
        if file_hash(trace_path) != result["trace_sha256"]:
            errors.append(f"trace hash mismatch: {status_row['attempt_id']}")
            continue
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != max_steps:
            errors.append(f"trace length mismatch: {status_row['attempt_id']}")
        for expected_step, row in enumerate(rows):
            if row["step"] != expected_step:
                errors.append(f"non-contiguous trace step: {status_row['attempt_id']}")
                break
            if row["label_available_at"] <= row["step"]:
                errors.append(f"label timing violation: {status_row['attempt_id']}")
                break
            numeric = (row["prediction"], row["target"], row["squared_error"])
            if any(not math.isfinite(float(value)) for value in numeric):
                errors.append(f"non-finite trace value: {status_row['attempt_id']}")
                break
        usage = result["metrics"]["resource_usage"]
        if usage["steps"] != max_steps:
            errors.append(f"usage step mismatch: {status_row['attempt_id']}")
        if usage["peak_state_bytes"] > max_state_bytes:
            errors.append(f"state-byte budget violation: {status_row['attempt_id']}")
        if not (attempt_dir / "checkpoint.json").is_file():
            errors.append(f"checkpoint missing: {status_row['attempt_id']}")
        verified_attempts += 1
        verified_trace_rows += len(rows)

    actual_failed = sum(row["status"] == "FAILED" for row in status_rows)
    actual_succeeded = sum(row["status"] == "SUCCEEDED" for row in status_rows)
    if receipt["failed"] != actual_failed or receipt["succeeded"] != actual_succeeded:
        errors.append("receipt status totals mismatch")

    artifact_hashes = {
        path.relative_to(bundle).as_posix(): file_hash(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "verification.json"
    }
    return {
        "schema_version": "neurogenesis.bundle-verification.v1",
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "verified_attempts": verified_attempts,
        "verified_trace_rows": verified_trace_rows,
        "protected_evaluation_runs": 0,
        "artifact_count": len(artifact_hashes),
        "artifact_hashes": artifact_hashes,
        "artifact_map_sha256": json_hash(artifact_hashes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verification receipt: {args.output}")
    result = verify(args.root.resolve(), args.bundle.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "artifact_hashes"}, indent=2))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())

