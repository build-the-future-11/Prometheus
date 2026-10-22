"""Aggregate every retained attempt, including failures, without outcome filtering."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .contracts import ContractError


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_bundle(bundle: Path) -> dict[str, Any]:
    result_paths = sorted((bundle / "attempts").glob("*/result.json"))
    if not result_paths:
        raise ContractError(f"no retained result files found under {bundle}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_policy.setdefault(result["policy_name"], []).append(result)

    summaries: dict[str, Any] = {}
    for policy, group in sorted(by_policy.items()):
        successes = [result for result in group if result["status"] == "SUCCEEDED"]
        primary = [result["metrics"]["primary"]["value"] for result in successes]
        adaptation = [result["metrics"]["adaptation_switch_mse"] for result in successes]
        retention = [result["metrics"]["retention_return_mse"] for result in successes]
        walls = [result["metrics"]["resource_usage"]["wall_seconds"] for result in successes]
        bytes_used = [
            result["metrics"]["resource_usage"]["peak_state_bytes"] for result in successes
        ]
        updates = [
            result["metrics"]["resource_usage"]["parameter_updates"] for result in successes
        ]
        summaries[policy] = {
            "attempted": len(group),
            "succeeded": len(successes),
            "failed": len(group) - len(successes),
            "primary_mse_mean": _mean(primary),
            "primary_mse_sample_stdev": _stdev(primary),
            "adaptation_switch_mse_mean": _mean(adaptation),
            "retention_return_mse_mean": _mean(retention),
            "wall_seconds_mean": _mean(walls),
            "peak_state_bytes_max": max(bytes_used) if bytes_used else None,
            "parameter_updates_total": sum(updates),
            "failed_attempt_ids": [
                result["attempt_id"] for result in group if result["status"] == "FAILED"
            ],
        }

    baseline_name = "no_adaptation"
    baseline_by_seed = {
        result["seed"]: result["metrics"]["primary"]["value"]
        for result in by_policy.get(baseline_name, [])
        if result["status"] == "SUCCEEDED"
    }
    paired: dict[str, Any] = {}
    for policy, group in sorted(by_policy.items()):
        if policy == baseline_name:
            continue
        differences = [
            result["metrics"]["primary"]["value"] - baseline_by_seed[result["seed"]]
            for result in group
            if result["status"] == "SUCCEEDED" and result["seed"] in baseline_by_seed
        ]
        paired[policy] = {
            "comparison": "policy_mse_minus_no_adaptation_mse",
            "paired_seed_count": len(differences),
            "mean_difference": _mean(differences),
            "sample_stdev": _stdev(differences),
        }

    successful_standard = {
        name: summary
        for name, summary in summaries.items()
        if name != baseline_name and summary["failed"] == 0
    }
    selected = min(
        successful_standard,
        key=lambda name: successful_standard[name]["primary_mse_mean"],
    ) if successful_standard else None
    return {
        "schema_version": "neurogenesis.aggregate.v1",
        "bundle": bundle.as_posix(),
        "attempt_count": len(results),
        "failure_count": sum(result["status"] == "FAILED" for result in results),
        "policy_summaries": summaries,
        "paired_against_no_adaptation": paired,
        "development_selected_standard_baseline": selected,
        "claim_boundary": (
            "Development-only baseline characterization; no candidate mechanism or protected "
            "evaluation was run."
        ),
    }


def write_aggregate(bundle: Path, output_dir: Path) -> dict[str, Any]:
    aggregate = aggregate_bundle(bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    columns = [
        "policy",
        "attempted",
        "succeeded",
        "failed",
        "primary_mse_mean",
        "primary_mse_sample_stdev",
        "adaptation_switch_mse_mean",
        "retention_return_mse_mean",
        "wall_seconds_mean",
        "peak_state_bytes_max",
        "parameter_updates_total",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for policy, summary in aggregate["policy_summaries"].items():
            writer.writerow({"policy": policy, **{key: summary[key] for key in columns[1:]}})
    return aggregate

