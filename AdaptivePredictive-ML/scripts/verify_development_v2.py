#!/usr/bin/env python3
"""Independently verify the retained Kaggle development-v2 evidence.

This verifier deliberately imports only the Python standard library and does
not import the research package.  It recomputes metrics and registered point
contrasts from the raw prediction ledger, validates denominators and hashes,
and writes a machine-readable receipt alongside the downloaded artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any


METHODS = (
    "seasonal_naive",
    "uniform",
    "hedge",
    "adahedge",
    "fixed_share",
    "adaptive_share",
    "directed_fixed_share",
    "ddfs",
)
CONTRASTS = {
    "ddfs_minus_fixed_share": ("ddfs", "fixed_share"),
    "ddfs_minus_adaptive_share": ("ddfs", "adaptive_share"),
    "ddfs_minus_directed_fixed_share": ("ddfs", "directed_fixed_share"),
    "directed_fixed_share_minus_fixed_share": ("directed_fixed_share", "fixed_share"),
    "adaptive_share_minus_fixed_share": ("adaptive_share", "fixed_share"),
    "ddfs_minus_adahedge": ("ddfs", "adahedge"),
}
TOLERANCE = 1e-12


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank prediction row at line {line_number}")
            row = json.loads(
                line,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
            if not isinstance(row, dict):
                raise TypeError(f"prediction row {line_number} is not an object")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def require(condition: bool, issue: str, issues: list[str]) -> None:
    if not condition:
        issues.append(issue)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    run_root = artifact_root / "kaggle-development-v2"
    report_path = artifact_root / "independent-verification-v2.json"
    issues: list[str] = []

    checksums = load_json(run_root / "checksums.json")
    observed_hashes = {
        name: sha256(run_root / name)
        for name in ("manifest.json", "predictions.jsonl", "summary.json")
    }
    require(observed_hashes == checksums, "inner-checksum-mismatch", issues)

    archive_hash = sha256(artifact_root / "kaggle-development-v2.tar.gz")
    recorded_archive_hash = (artifact_root / "kaggle-development-v2.sha256").read_text().split()[0]
    require(archive_hash == recorded_archive_hash, "archive-checksum-mismatch", issues)

    manifest = load_json(run_root / "manifest.json")
    summary = load_json(run_root / "summary.json")
    analysis = load_json(artifact_root / "development-analysis-v2.json")
    rows = load_jsonl(run_root / "predictions.jsonl")
    require(manifest.get("schema_version") == "adaptive-predictive.run.v1", "schema-mismatch", issues)
    require(manifest.get("stage") == "development-v2", "stage-mismatch", issues)

    specs = {item["series"]: item for item in manifest["series"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("series")].append(row)
    require(set(grouped) == set(specs), "series-set-mismatch", issues)
    require(set(summary) == set(specs), "summary-series-set-mismatch", issues)

    recomputed: dict[str, dict[str, dict[str, float]]] = {}
    scales: dict[str, float] = {}
    point_contrasts: dict[str, dict[str, float]] = {name: {} for name in CONTRASTS}
    for series, spec in sorted(specs.items()):
        series_rows = grouped[series]
        expected_count = int(spec["n_evaluated"])
        require(len(series_rows) == expected_count, f"row-count:{series}", issues)
        expected_index = int(spec["evaluate_from"])
        start = date.fromisoformat(series_rows[0]["timestamp"])
        for offset, row in enumerate(series_rows):
            require(row["index"] == expected_index + offset, f"index-sequence:{series}:{offset}", issues)
            require(
                date.fromisoformat(row["timestamp"]) == start + timedelta(days=offset),
                f"timestamp-sequence:{series}:{offset}",
                issues,
            )
            require(set(row["forecast"]) == set(METHODS), f"method-set:{series}:{offset}", issues)
            require(
                set(row["diagnostics"]) == set(METHODS) - {"seasonal_naive"},
                f"diagnostics-set:{series}:{offset}",
                issues,
            )

        # Recover the training-only MASE scale from two independently recorded
        # summary fields, then demand that every method yields the same scale.
        candidate_scales = []
        for method in METHODS:
            item = summary[series][method]
            candidate_scales.append(float(item["mae"]) / float(item["mase"]))
        scale = candidate_scales[0]
        require(
            all(close(value, scale) for value in candidate_scales),
            f"mase-scale-inconsistent:{series}",
            issues,
        )
        scales[series] = scale

        recomputed[series] = {}
        for method in METHODS:
            errors = [float(row["forecast"][method]) - float(row["outcome"]) for row in series_rows]
            metrics = {
                "mae": mean([abs(value) for value in errors]),
                "mase": mean([abs(value) for value in errors]) / scale,
                "rmse": math.sqrt(mean([value * value for value in errors])),
                "bias": mean(errors),
            }
            if method != "seasonal_naive":
                diagnostics = [row["diagnostics"][method] for row in series_rows]
                metrics["interval_coverage"] = mean([
                    float(item["lower"] <= row["outcome"] <= item["upper"])
                    for item, row in zip(diagnostics, series_rows, strict=True)
                ])
                metrics["mean_interval_width"] = mean([
                    float(item["upper"]) - float(item["lower"]) for item in diagnostics
                ])
            recomputed[series][method] = metrics
            for metric, value in metrics.items():
                expected = float(summary[series][method][metric])
                require(close(value, expected), f"metric:{series}:{method}:{metric}", issues)

        for contrast_name, (left, right) in CONTRASTS.items():
            value = mean([
                (
                    abs(float(row["forecast"][left]) - float(row["outcome"]))
                    - abs(float(row["forecast"][right]) - float(row["outcome"]))
                ) / scale
                for row in series_rows
            ])
            point_contrasts[contrast_name][series] = value
            expected = float(analysis["contrasts"][contrast_name]["series_differences"][series])
            require(close(value, expected), f"series-contrast:{contrast_name}:{series}", issues)

    aggregate_mase = {
        method: mean([recomputed[series][method]["mase"] for series in sorted(recomputed)])
        for method in METHODS
    }
    for method, value in aggregate_mase.items():
        require(
            close(value, float(analysis["mean_series_weighted_mase"][method])),
            f"aggregate-mase:{method}",
            issues,
        )

    aggregate_contrasts = {}
    for name, values in point_contrasts.items():
        estimate = mean(list(values.values()))
        wins = sum(value < 0.0 for value in values.values())
        aggregate_contrasts[name] = {"estimate": estimate, "series_wins": wins}
        reported = analysis["contrasts"][name]
        require(close(estimate, float(reported["bootstrap"]["estimate"])), f"contrast-estimate:{name}", issues)
        require(wins == int(reported["series_wins"]), f"contrast-wins:{name}", issues)

    relative_improvement = (
        aggregate_mase["fixed_share"] - aggregate_mase["ddfs"]
    ) / aggregate_mase["fixed_share"]
    require(
        close(relative_improvement, float(analysis["relative_improvement_vs_fixed_share"])),
        "relative-improvement",
        issues,
    )
    primary_gate = (
        aggregate_contrasts["ddfs_minus_fixed_share"]["estimate"] <= -0.01
        and relative_improvement >= 0.01
    )
    directed_gate = (
        aggregate_contrasts["ddfs_minus_adaptive_share"]["estimate"] < 0.0
        and aggregate_contrasts["ddfs_minus_adaptive_share"]["series_wins"] * 2 >= len(specs)
    )
    require(primary_gate is analysis["registered_primary_threshold_met"], "primary-gate", issues)
    require(directed_gate is analysis["directed_mechanism_development_gate_met"], "directed-gate", issues)

    report = {
        "schema_version": "adaptive-predictive.independent-verification.v1",
        "valid": not issues,
        "issues": issues,
        "method": "stdlib-only recomputation from raw predictions; no research-package imports",
        "artifact_root": str(artifact_root),
        "archive_sha256": archive_hash,
        "inner_sha256": observed_hashes,
        "prediction_rows": len(rows),
        "series_count": len(specs),
        "series_denominator": {name: len(grouped[name]) for name in sorted(grouped)},
        "recovered_mase_scale": scales,
        "mean_series_weighted_mase": aggregate_mase,
        "point_contrasts": aggregate_contrasts,
        "relative_improvement_vs_fixed_share": relative_improvement,
        "registered_primary_threshold_met": primary_gate,
        "directed_mechanism_development_gate_met": directed_gate,
        "bootstrap_scope": "Bootstrap implementation is covered by hosted tests; CI values are retained, not independently regenerated here.",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
