#!/usr/bin/env python3
"""Independently verify a retained DLML-S3 Kaggle output archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


class VerificationError(RuntimeError):
    """Raised when hosted evidence violates the frozen verification contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def strict_json_bytes(payload: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VerificationError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise VerificationError(f"non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON in {label}: {error}") from error


def canonical_parameters(parameters: Any) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)


def classify(
    paired_count: int,
    required_count: int,
    mean_improvement: float | None,
    wins: int,
    minimum_improvement: float,
    minimum_wins: int,
) -> str:
    if paired_count != required_count or mean_improvement is None:
        return "INCONCLUSIVE"
    improvement_met = mean_improvement >= minimum_improvement
    wins_met = wins >= minimum_wins
    if improvement_met and wins_met:
        return "SUPPORTED"
    if improvement_met or wins_met:
        return "MIXED"
    return "NEGATIVE"


def close_float(actual: Any, expected: Any, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise VerificationError(f"{label} mismatch: {actual!r} != {expected!r}")
        return
    if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
        raise VerificationError(f"{label} is not finite numeric evidence: {actual!r}")
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-9):
        raise VerificationError(f"{label} mismatch: {actual!r} != {expected!r}")


def parse_sidecar(sidecar: Path, archive: Path) -> str:
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != archive.name:
        raise VerificationError("sidecar must contain exactly '<sha256>  <archive-name>'")
    digest = fields[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise VerificationError("sidecar digest is not lowercase SHA-256")
    actual = sha256_file(archive)
    if actual != digest:
        raise VerificationError(f"archive SHA-256 mismatch: {actual} != {digest}")
    return actual


def read_archive(archive_path: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise VerificationError(f"unsafe archive path: {member.name}")
            if member.name in payloads:
                raise VerificationError(f"duplicate archive member: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise VerificationError(f"unsupported archive member type: {member.name}")
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise VerificationError(f"unreadable archive member: {member.name}")
                payloads[member.name] = stream.read()
    return payloads


def require_member(payloads: dict[str, bytes], name: str) -> bytes:
    if name not in payloads:
        raise VerificationError(f"required archive member is missing: {name}")
    return payloads[name]


def read_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    lines = payload.splitlines()
    if not lines:
        raise VerificationError(f"empty JSONL evidence: {label}")
    rows = [strict_json_bytes(line, f"{label}:{index}") for index, line in enumerate(lines, start=1)]
    if not all(isinstance(row, dict) for row in rows):
        raise VerificationError(f"non-object JSONL row in {label}")
    return rows


def verify_archive(
    archive_path: Path,
    sidecar_path: Path,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    archive_sha256 = parse_sidecar(sidecar_path, archive_path)
    payloads = read_archive(archive_path)
    manifests = sorted(name for name in payloads if name.startswith("runs/") and name.endswith("/manifest.json"))
    if len(manifests) != 1:
        raise VerificationError(f"expected exactly one run manifest, found {manifests}")
    manifest_name = manifests[0]
    run_root = manifest_name.removesuffix("/manifest.json")
    manifest = strict_json_bytes(payloads[manifest_name], manifest_name)
    config = strict_json_bytes(config_path.read_bytes(), str(config_path))
    protocol_path = project_root / config["protocol_path"]

    if manifest.get("status") != "COMPLETE":
        raise VerificationError(f"run manifest is not COMPLETE: {manifest.get('status')!r}")
    if manifest.get("study_id") != config.get("study_id"):
        raise VerificationError("manifest study_id does not match the frozen config")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise VerificationError("manifest config hash does not match the frozen config")
    if manifest.get("protocol_sha256") != sha256_file(protocol_path):
        raise VerificationError("manifest protocol hash does not match the frozen protocol")

    declared_outputs = manifest.get("outputs")
    if not isinstance(declared_outputs, dict) or not declared_outputs:
        raise VerificationError("manifest outputs mapping is missing or empty")
    for relative, expected_hash in declared_outputs.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise VerificationError(f"unsafe manifest output path: {relative}")
        member = f"{run_root}/{relative}"
        actual_hash = sha256_bytes(require_member(payloads, member))
        if actual_hash != expected_hash:
            raise VerificationError(f"manifest hash mismatch for {relative}: {actual_hash} != {expected_hash}")

    archived_run_files = {
        name.removeprefix(f"{run_root}/")
        for name in payloads
        if name.startswith(f"{run_root}/") and name != manifest_name
    }
    if archived_run_files != set(declared_outputs):
        missing = sorted(set(declared_outputs) - archived_run_files)
        extra = sorted(archived_run_files - set(declared_outputs))
        raise VerificationError(f"run archive/manifest membership mismatch; missing={missing}, extra={extra}")

    attempts = read_jsonl(require_member(payloads, f"{run_root}/selection_attempts.jsonl"), "selection_attempts.jsonl")
    results = read_jsonl(require_member(payloads, f"{run_root}/results.jsonl"), "results.jsonl")
    selection = strict_json_bytes(require_member(payloads, f"{run_root}/selection.json"), "selection.json")
    summary = strict_json_bytes(require_member(payloads, f"{run_root}/summary.json"), "summary.json")
    strict_json_bytes(require_member(payloads, f"{run_root}/environment.json"), "environment.json")
    strict_json_bytes(require_member(payloads, f"{run_root}/resource_admission.json"), "resource_admission.json")

    seeds = [int(seed) for seed in config["seeds"]]
    methods = list(config["candidate_grid"])
    expected_attempts = {
        (seed, method, canonical_parameters(parameters))
        for seed in seeds
        for method, candidates in config["candidate_grid"].items()
        for parameters in candidates
    }
    actual_attempts: set[tuple[int, str, str]] = set()
    successful_candidates: set[tuple[int, str, str]] = set()
    failed_attempts = 0
    for row in attempts:
        key = (int(row.get("seed")), row.get("method"), canonical_parameters(row.get("parameters")))
        if key in actual_attempts:
            raise VerificationError(f"duplicate selection attempt: {key}")
        actual_attempts.add(key)
        if row.get("test_metrics_present") is not False or "test" in row:
            raise VerificationError(f"selection attempt contains protected-test evidence: {key}")
        status = row.get("status")
        if status == "SUCCEEDED":
            if "development" not in row:
                raise VerificationError(f"successful selection attempt lacks development metrics: {key}")
            successful_candidates.add(key)
        elif status == "FAILED":
            failed_attempts += 1
            if not row.get("error") or not row.get("traceback"):
                raise VerificationError(f"failed selection attempt lacks error/traceback: {key}")
        else:
            raise VerificationError(f"non-terminal selection status {status!r}: {key}")
    if actual_attempts != expected_attempts:
        raise VerificationError("selection-attempt matrix does not match the frozen candidate grid")

    if selection.get("uses_test_targets") is not False:
        raise VerificationError("selection does not explicitly deny protected-test target use")
    selected = selection.get("selected")
    if not isinstance(selected, dict) or set(selected) != {str(seed) for seed in seeds}:
        raise VerificationError("selection seed set does not match the frozen config")
    for seed in seeds:
        if set(selected[str(seed)]) != set(methods):
            raise VerificationError(f"selected method set is incomplete for seed {seed}")
        for method in methods:
            key = (seed, method, canonical_parameters(selected[str(seed)][method]))
            if key not in successful_candidates:
                raise VerificationError(f"selected candidate was not a successful attempt: {key}")

    expected_results = {(seed, method) for seed in seeds for method in methods}
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    failed_results = 0
    for row in results:
        key = (int(row.get("seed")), row.get("method"))
        if key in by_key:
            raise VerificationError(f"duplicate protected-test result: {key}")
        by_key[key] = row
        if row.get("split") != "protected_test":
            raise VerificationError(f"result has wrong split: {key}")
        if canonical_parameters(row.get("parameters")) != canonical_parameters(selected[str(key[0])][key[1]]):
            raise VerificationError(f"protected-test parameters differ from locked selection: {key}")
        status = row.get("status")
        if status == "SUCCEEDED":
            test = row.get("test")
            if not isinstance(test, dict):
                raise VerificationError(f"successful result lacks test metrics: {key}")
            for metric in ("standardized_rmse", "rmse", "wall_seconds", "peak_gpu_bytes"):
                value = test.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                    raise VerificationError(f"invalid {metric} for {key}: {value!r}")
        elif status == "FAILED":
            failed_results += 1
            if not row.get("error") or not row.get("traceback"):
                raise VerificationError(f"failed result lacks error/traceback: {key}")
        else:
            raise VerificationError(f"non-terminal result status {status!r}: {key}")
    if set(by_key) != expected_results:
        raise VerificationError("protected-test result matrix does not match the frozen seed/method grid")

    paired: list[dict[str, Any]] = []
    for seed in seeds:
        joint = by_key[(seed, "joint_block_mlp")]
        residual = by_key[(seed, "residual_block_mlp")]
        if joint["status"] == residual["status"] == "SUCCEEDED":
            joint_rmse = float(joint["test"]["rmse"])
            residual_rmse = float(residual["test"]["rmse"])
            paired.append(
                {
                    "seed": seed,
                    "joint_rmse": joint_rmse,
                    "residual_rmse": residual_rmse,
                    "relative_improvement": (joint_rmse - residual_rmse) / joint_rmse,
                }
            )
    mean_improvement = statistics.fmean(row["relative_improvement"] for row in paired) if paired else None
    wins = sum(row["relative_improvement"] > 0 for row in paired)
    criteria = config["success_criteria"]
    registered_result = classify(
        len(paired),
        len(seeds),
        mean_improvement,
        wins,
        float(criteria["minimum_relative_rmse_improvement"]),
        int(criteria["minimum_paired_wins"]),
    )
    method_means: dict[str, float | None] = {}
    for method in methods:
        scores = [float(by_key[(seed, method)]["test"]["rmse"]) for seed in seeds if by_key[(seed, method)]["status"] == "SUCCEEDED"]
        method_means[method] = statistics.fmean(scores) if scores else None

    if summary.get("study_id") != config["study_id"]:
        raise VerificationError("summary study_id does not match the frozen config")
    if summary.get("paired_count") != len(paired) or summary.get("paired_wins") != wins:
        raise VerificationError("summary paired count/wins do not match raw results")
    close_float(summary.get("mean_relative_improvement"), mean_improvement, "mean_relative_improvement")
    if summary.get("registered_result") != registered_result:
        raise VerificationError("summary terminal classification does not match raw results")
    if summary.get("broad_superiority_claim_allowed") is not False:
        raise VerificationError("summary improperly permits a broad superiority claim")
    if summary.get("independent_reproduction") is not False:
        raise VerificationError("summary improperly claims independent reproduction")
    summary_means = summary.get("method_mean_rmse")
    if not isinstance(summary_means, dict) or set(summary_means) != set(methods):
        raise VerificationError("summary method means are incomplete")
    for method, expected in method_means.items():
        close_float(summary_means[method], expected, f"method_mean_rmse[{method}]")
    summary_paired = summary.get("paired")
    if not isinstance(summary_paired, list) or len(summary_paired) != len(paired):
        raise VerificationError("summary paired rows do not match raw results")
    for actual, expected in zip(summary_paired, paired):
        if actual.get("seed") != expected["seed"]:
            raise VerificationError("summary paired seed order differs from the frozen order")
        for field in ("joint_rmse", "residual_rmse", "relative_improvement"):
            close_float(actual.get(field), expected[field], f"paired[{expected['seed']}].{field}")

    split_receipts = sorted(name for name in payloads if name.startswith(f"{run_root}/splits/seed-") and name.endswith(".json"))
    expected_split_receipts = {f"{run_root}/splits/seed-{seed}.json" for seed in seeds}
    if set(split_receipts) != expected_split_receipts:
        raise VerificationError("split receipt set does not match the frozen seeds")
    for name in split_receipts:
        split = strict_json_bytes(payloads[name], name)
        if split.get("test_targets_used_for_selection") is not False:
            raise VerificationError(f"split receipt does not deny test-target selection use: {name}")

    return {
        "schema_version": "derived-learning.s3-hosted-verification.v1",
        "verification_status": "VERIFIED",
        "study_id": config["study_id"],
        "archive_sha256": archive_sha256,
        "manifest_sha256": sha256_bytes(payloads[manifest_name]),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "selection_attempt_count": len(attempts),
        "selection_failure_count": failed_attempts,
        "protected_result_count": len(results),
        "protected_failure_count": failed_results,
        "paired_count": len(paired),
        "paired_wins": wins,
        "mean_relative_improvement": mean_improvement,
        "method_mean_rmse": method_means,
        "registered_result": registered_result,
        "broad_superiority_claim_allowed": False,
        "independent_reproduction": False,
    }


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/s3_real_data_gpu.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        receipt = verify_archive(
            arguments.archive.resolve(),
            arguments.sidecar.resolve(),
            arguments.config.resolve(),
            arguments.project_root.resolve(),
        )
        exit_code = 0
    except Exception as error:
        receipt = {
            "schema_version": "derived-learning.s3-hosted-verification.v1",
            "verification_status": "INVALID",
            "registered_result": None,
            "error": f"{type(error).__name__}: {error}",
            "broad_superiority_claim_allowed": False,
            "independent_reproduction": False,
        }
        exit_code = 1
    write_receipt(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
