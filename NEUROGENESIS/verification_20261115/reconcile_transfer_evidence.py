"""Check frozen method coverage and independently recompute transfer claim fields.

This additive audit lives outside the frozen source set. It does not execute or
modify the predictor. It supplements the frozen exact replay and arithmetic
auditor by binding the manifest to the original lock and checking the top-level
decision fields that the original transfer verifier does not recompute.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path

METHODS = frozenset({"all_update", "baseline", "candidate", "capacity_one",
                     "frozen", "knn", "no_bank", "periodic", "replay",
                     "uniform_bank", "last_observation"})


def load(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject_constant,
                      object_pairs_hook=unique)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_number(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual), f"Invalid numeric field: {label}")
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15),
            f"Numeric mismatch: {label}: {actual} != {expected}")


def check_decision(summary, methods):
    complete = all(row["status"] == "COMPLETED" for row in methods.values())
    require(type(summary.get("all_completed")) is bool
            and summary["all_completed"] == complete, "Wrong all_completed field")
    if not complete:
        require(summary.get("descriptive_support") is False,
                "Failed methods cannot support a comparison")
        require(summary.get("relative_mse_change") is None
                and summary.get("update_reduction") is None,
                "Failed comparison must not expose a selected valid-only effect")
        return {"disposition": "INCONCLUSIVE", "failed_methods": [
            name for name, row in methods.items() if row["status"] != "COMPLETED"]}
    candidate, baseline = methods["candidate"], methods["baseline"]
    effect = candidate["test_mse_scaled"] / baseline["test_mse_scaled"] - 1
    reduction = (1 - candidate["test_updates"] / baseline["test_updates"]
                 if baseline["test_updates"] else None)
    support = effect <= 0.02 and reduction is not None and reduction >= 0.20
    require(type(summary.get("descriptive_support")) is bool
            and summary["descriptive_support"] == support,
            "Wrong descriptive_support field")
    equal_number(summary.get("relative_mse_change"), effect, "relative_mse_change")
    if reduction is None:
        require(summary.get("update_reduction") is None, "Invalid update reduction")
    else:
        equal_number(summary.get("update_reduction"), reduction, "update_reduction")
    return {"disposition": "SUPPORTED_DESCRIPTIVE" if support else "NEGATIVE_DESCRIPTIVE",
            "relative_mse_change": effect, "weight_solve_reduction": reduction,
            "candidate_vs_last_observation_mse_ratio": (
                candidate["test_mse_scaled"] / methods["last_observation"]["test_mse_scaled"])}


def reconcile(source: Path, run: Path, execution_receipt: Path):
    lock = load(source / "runs" / "freeze.json")
    manifest = load(run / "manifest.json")
    summary = load(run / "summary.json")
    expected_configs = dict(lock["configs"])
    expected_configs["last_observation"] = {"kind": "last"}
    require(set(expected_configs) == METHODS, "Unexpected frozen method roster")
    require(manifest["configs"] == expected_configs, "Manifest differs from original lock")
    require(set(summary["results"]) == METHODS, "Missing or extra summary methods")
    require(manifest["source"] == lock["source"], "Manifest source differs from lock")
    for name, expected in lock["source"].items():
        require(sha(source / name) == expected, f"Frozen source drift: {name}")
    require(load(source / "protocol.json") == lock["protocol"], "Protocol differs from lock")
    require(manifest["lock_sha256"] == canonical_digest(lock), "Lock digest mismatch")
    dataset = manifest["dataset"]
    require(dataset in {"air", "beijing"}, "Unknown dataset")
    rows, boundary, scale = ((9357, 3742, 100.0) if dataset == "air"
                             else (17520, 8760, 10000.0))
    phase = "exposed_regression" if dataset == "air" else "protected_transfer"
    require(manifest["phase"] == phase and summary["phase"] == phase,
            "Frozen phase label mismatch")
    require((manifest["rows"], manifest["test_start"], manifest["scale_squared"])
            == (rows, boundary, scale), "Calendar/scale contract mismatch")
    data = load(source / "data" / "manifest.json")[dataset]
    require(manifest["data"] == data, "Data manifest mismatch")
    require(sha(source / "data" / data["file"]) == data["sha256"], "Dataset digest mismatch")
    admission = load(run / "admission.json")
    require(admission.get("allowed") is True, "Resource admission was not granted")
    before, after = admission["before"], admission["after"]
    require(before["platform"] == after["platform"], "Admission platform changed")
    require(after["rss"] < lock["protocol"]["process_rss_bytes"], "Admission RSS cap exceeded")
    if after["platform"] == "darwin":
        require(after["reclaimable_bytes"] >= 512 * 1024 ** 2
                and before["pageouts"] == after["pageouts"], "macOS resource admission failed")
    attempt_paths = list((run / "attempts").glob("*/result.json"))
    expected_paths = {f"{dataset}__{name}" for name in METHODS}
    require({path.parent.name for path in attempt_paths} == expected_paths
            and len(attempt_paths) == len(METHODS), "Missing, extra or duplicated attempts")
    records = {}
    maximum_mse_difference = 0.0
    actual_trace_rows = 0
    arithmetic_map = {}
    for name in sorted(METHODS):
        directory = run / "attempts" / f"{dataset}__{name}"
        result = load(directory / "result.json")
        saved = summary["results"][name]
        require(result["method"] == name and result["task_id"] == dataset,
                f"Attempt identity mismatch: {name}")
        require(result["status"] in {"COMPLETED", "FAILED"}, f"Unknown status: {name}")
        require(result["status"] == saved["status"], f"Summary status mismatch: {name}")
        for artifact in ("trace", "checkpoint"):
            suffix = ".jsonl.gz" if artifact == "trace" else ".json.gz"
            require(sha(directory / (artifact + suffix)) == result[artifact + "_sha256"],
                    f"Artifact digest mismatch: {name}/{artifact}")
            arithmetic_map[str((directory / (artifact + suffix)).relative_to(run))] = sha(
                directory / (artifact + suffix))
        require(result["process_peak_rss_bytes"] <= lock["protocol"]["process_rss_bytes"],
                f"RSS cap exceeded: {name}")
        if result["status"] == "FAILED":
            require(bool(result["failure"]), f"Missing failure receipt: {name}")
            with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
                actual_trace_rows += sum(1 for _ in stream)
            records[name] = {"status": "FAILED", "failure": result["failure"]}
            continue
        with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
            trace = [json.loads(line) for line in stream]
        actual_trace_rows += len(trace)
        require(len(trace) == rows, f"Incomplete trace: {name}")
        errors = [(event["prediction"] - event["target"]) ** 2
                  for event in trace[boundary:] if event["target"] is not None]
        require(bool(errors) and all(math.isfinite(error) for error in errors),
                f"No valid evaluation errors: {name}")
        mse = math.fsum(errors) / len(errors)
        usage = {key: trace[-1]["usage"][key] - trace[boundary - 1]["usage"][key]
                 for key in trace[-1]["usage"]}
        equal_number(saved["test_mse_scaled"], mse, f"{name} scaled MSE")
        equal_number(saved["test_mse_original_units"], mse * scale, f"{name} original MSE")
        require(saved["test_scored"] == len(errors), f"Score denominator mismatch: {name}")
        require(saved["test_usage"] == usage, f"Evaluation counters mismatch: {name}")
        require(saved["test_updates"] == usage["updates"], f"Solve count mismatch: {name}")
        require(max(event["state_bytes"] for event in trace)
                <= lock["protocol"]["algorithmic_state_bytes"], f"State cap exceeded: {name}")
        maximum_mse_difference = max(maximum_mse_difference, abs(mse - saved["test_mse_scaled"]))
        records[name] = {"status": "COMPLETED", "test_mse_scaled": mse,
                         "test_mse_original_units": mse * scale,
                         "test_scored": len(errors), "test_updates": usage["updates"],
                         "process_peak_rss_bytes": result["process_peak_rss_bytes"],
                         "peak_state_bytes": result["peak_state_bytes"],
                         "producer_wall_seconds": result["wall_seconds"]}
    decision = check_decision(summary, records)
    exact, arithmetic = load(run / "verification-exact.json"), load(run / "audit.json")
    require(exact["status"] == "VALID" and exact["exact_replay"] is True,
            "Missing valid exact replay")
    require(exact["methods"] == len(METHODS), "Exact replay denominator mismatch")
    require(exact["verified_trace_rows"] == actual_trace_rows, "Exact replay row count mismatch")
    require(arithmetic["status"] == "VALID" and arithmetic["attempts"] == len(METHODS),
            "Missing valid arithmetic audit")
    arithmetic_digest = hashlib.sha256(json.dumps(arithmetic_map, sort_keys=True).encode()).hexdigest()
    require(arithmetic.get("artifact_map_sha256") == arithmetic_digest,
            "Arithmetic audit digest does not match current trace/checkpoint artifacts")
    failed_methods = {name for name, row in records.items() if row["status"] == "FAILED"}
    require(exact["failed"] == len(failed_methods), "Exact replay failure count mismatch")
    require({tuple(item) for item in arithmetic["failed"]}
            == {(dataset, name) for name in failed_methods}, "Arithmetic failure roster mismatch")
    if decision["disposition"] != "INCONCLUSIVE":
        require(exact["failed"] == 0 and arithmetic["failed"] == [], "Failures hidden in receipt")
        require(exact["verified_trace_rows"] == rows * len(METHODS), "Replay row count mismatch")
    wall_sum = math.fsum(row.get("producer_wall_seconds", 0) for row in records.values())
    require(wall_sum <= lock["protocol"]["phase_cpu_wall_seconds"], "Recorded producer budget exceeded")
    execution = load(execution_receipt)
    require(execution.get("all_commands_succeeded") is True
            and execution.get("frozen_source_unchanged") is True,
            "Execution wrapper did not finish successfully with frozen source")
    require(execution["dataset"] == dataset
            and execution["frozen_source_sha256"] == lock["source"]
            and execution["freeze_sha256"] == sha(source / "runs" / "freeze.json"),
            "Execution receipt source, dataset or freeze identity differs")
    commands = execution["commands"]
    require(len(commands) == 3 and all(item["returncode"] == 0 for item in commands),
            "Execution receipt lacks three successful commands")
    require(commands[0]["argv"][1:3] == ["transfer.py", "run"]
            and commands[1]["argv"][1:3] == ["transfer.py", "verify"]
            and "--replay" in commands[1]["argv"]
            and commands[2]["argv"][1] == "audit.py", "Unexpected executed workflow")
    require(commands[0]["wall_seconds"] <= lock["protocol"]["phase_cpu_wall_seconds"],
            "Full producer command exceeded phase budget")
    for index, item in enumerate(commands, 1):
        require(sha(execution_receipt.parent / f"command-{index}.log") == item["log_sha256"],
                f"Executed command log changed: {index}")
    artifact_map = {str(path.relative_to(run)): sha(path)
                    for path in sorted(run.rglob("*")) if path.is_file()}
    require(execution["run_artifact_sha256"] == artifact_map,
            "Run artifacts differ from the original verified execution receipt")
    return {"status": "VALID", "schema": "neurogenesis.transfer-reconciliation.v1",
            "dataset": dataset, "scope": "Frozen-code evidence reconstruction; historical outcomes already exposed",
            "new_protected_confirmation": False, "independent_external_reproduction": False,
            "methods": records, "decision": decision,
            "trace_rows": actual_trace_rows, "expected_complete_trace_rows": rows * len(METHODS),
            "maximum_fsum_mse_difference": maximum_mse_difference,
            "recorded_producer_wall_sum_seconds": wall_sum,
            "source_sha256": lock["source"], "freeze_sha256": sha(source / "runs" / "freeze.json"),
            "artifact_sha256": artifact_map,
            "execution_receipt_sha256": sha(execution_receipt),
            "arithmetic_artifact_map_sha256": arithmetic_digest,
            "utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.source, args.run, args.execution_receipt)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "dataset", "decision", "trace_rows")}))


if __name__ == "__main__":
    main()
