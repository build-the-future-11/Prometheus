"""Saved-output statistical summaries and dependency-free SVG figures."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _last_at_or_below(result: dict[str, Any], budget: int) -> dict[str, Any]:
    eligible = [point for point in result["checkpoints"] if int(point["arithmetic_work_units"]) <= budget]
    return eligible[-1] if eligible else result["checkpoints"][0]


def paired_bootstrap_mean_ci(values: list[float], *, samples: int = 10_000, seed: int = 20261020) -> dict[str, float | int]:
    """Deterministic percentile bootstrap over paired task deltas."""

    if not values:
        raise ValueError("paired bootstrap requires at least one value")
    rng = random.Random(seed)
    means = sorted(
        math.fsum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "bootstrap_95_ci_low": means[int(0.025 * (samples - 1))],
        "bootstrap_95_ci_high": means[int(0.975 * (samples - 1))],
    }


def _load_results(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Calculate secondary statistics from finalized, saved run data only."""

    results = [result for result in _load_results(run_dir) if result["status"] == "SUCCEEDED"]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_split[result["split"]].append(result)
    split_summaries: dict[str, Any] = {}
    for split, records in sorted(by_split.items()):
        by_pair = {(record["task_id"], record["method"]): record for record in records}
        methods = sorted({record["method"] for record in records})
        task_ids = sorted({record["task_id"] for record in records})
        paired_ids = [task_id for task_id in task_ids if all((task_id, method) in by_pair for method in methods)]
        if not paired_ids or "analytic_residual" not in methods or "adam" not in methods:
            split_summaries[split] = {"paired_task_count": 0, "status": "INSUFFICIENT_COMPARABLE_RESULTS"}
            continue
        deltas: list[float] = []
        for task_id in paired_ids:
            budget = min(int(by_pair[(task_id, method)]["checkpoints"][-1]["arithmetic_work_units"]) for method in methods)
            primary = float(_last_at_or_below(by_pair[(task_id, "analytic_residual")], budget)["evaluation_nmse"])
            baseline = float(_last_at_or_below(by_pair[(task_id, "adam")], budget)["evaluation_nmse"])
            deltas.append(primary - baseline)
        split_summaries[split] = {
            "paired_task_count": len(paired_ids),
            "primary_minus_adam_equal_resource_nmse": deltas,
            "paired_bootstrap_mean_delta": paired_bootstrap_mean_ci(deltas),
            "primary_win_count": sum(delta < 0.0 for delta in deltas),
            "analysis_scope": "descriptive paired bootstrap; not a replacement for the protocol primary endpoint",
        }
    return {
        "schema_version": "derived-learning.saved-output-analysis.v1",
        "run_dir": str(run_dir),
        "by_split": split_summaries,
    }


_COLORS = {
    "adam": "#0072B2",
    "analytic_residual": "#D55E00",
    "block_jacobi_residual": "#CC79A7",
    "block_gradient_no_derivation": "#009E73",
    "fixed_normalized_lms": "#E69F00",
    "classical_backfitting": "#56B4E9",
    "global_ridge": "#000000",
}


def _svg_curve(path: Path, split: str, records: list[dict[str, Any]]) -> None:
    """Render every recorded learning curve; no interpolation or best-point choice."""

    usable = [record for record in records if record["status"] == "SUCCEEDED" and record["checkpoints"]]
    if not usable:
        return
    width, height, left, top, right, bottom = 1120, 680, 100, 65, 35, 90
    max_x = max(int(point["arithmetic_work_units"]) for record in usable for point in record["checkpoints"])
    max_y = max(float(point["evaluation_nmse"]) for record in usable for point in record["checkpoints"])
    max_y = max(0.01, max_y * 1.05)

    def x(value: float) -> float:
        return left + value / max_x * (width - left - right) if max_x else left

    def y(value: float) -> float:
        return top + (1.0 - min(value, max_y) / max_y) * (height - top - bottom)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.axis{stroke:#202124;stroke-width:1.5}.grid{stroke:#d9d9d9;stroke-width:1}.curve{fill:none;stroke-width:1.4;opacity:.36}.legend{font-size:14px}.label{font-size:16px}.title{font-size:22px;font-weight:600}</style>',
        f'<text x="{left}" y="32" class="title">{split.replace("_", " ").title()} learning curves</text>',
    ]
    for tick in range(6):
        value = max_y * tick / 5
        y_value = y(value)
        elements.extend([
            f'<line x1="{left}" x2="{width-right}" y1="{y_value:.2f}" y2="{y_value:.2f}" class="grid"/>',
            f'<text x="{left-12}" y="{y_value+5:.2f}" text-anchor="end" class="legend">{value:.3g}</text>',
        ])
    for tick in range(6):
        value = max_x * tick / 5
        x_value = x(value)
        elements.extend([
            f'<line x1="{x_value:.2f}" x2="{x_value:.2f}" y1="{top}" y2="{height-bottom}" class="grid"/>',
            f'<text x="{x_value:.2f}" y="{height-bottom+28}" text-anchor="middle" class="legend">{value:.3g}</text>',
        ])
    elements.extend([
        f'<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" class="axis"/>',
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" class="axis"/>',
        f'<text x="{(left + width-right)/2:.2f}" y="{height-28}" text-anchor="middle" class="label">Accounted arithmetic work units</text>',
        f'<text x="24" y="{(top + height-bottom)/2:.2f}" text-anchor="middle" transform="rotate(-90 24 {(top + height-bottom)/2:.2f})" class="label">Evaluation NMSE</text>',
    ])
    methods = sorted({record["method"] for record in usable})
    for record in usable:
        coordinates = " ".join(
            f"{x(float(point['arithmetic_work_units'])):.2f},{y(float(point['evaluation_nmse'])):.2f}"
            for point in record["checkpoints"]
        )
        elements.append(f'<polyline points="{coordinates}" stroke="{_COLORS.get(record["method"], "#555555")}" class="curve"/>')
    for index, method in enumerate(methods):
        legend_x = left + index * 190
        color = _COLORS.get(method, "#555555")
        elements.extend([
            f'<line x1="{legend_x}" x2="{legend_x+28}" y1="{height-54}" y2="{height-54}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{legend_x+34}" y="{height-49}" class="legend">{method}</text>',
        ])
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_analysis(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    payload = analyze_run(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ANALYSIS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Saved-output statistical analysis", "", "This file is generated from finalized `results.jsonl`; it does not select, omit, or retune checkpoints.", ""]
    for split, values in payload["by_split"].items():
        lines.extend([f"## {split.replace('_', ' ').title()}", ""])
        if values.get("status"):
            lines.extend([f"Status: `{values['status']}`.", ""])
            continue
        interval = values["paired_bootstrap_mean_delta"]
        lines.extend([
            f"Paired tasks: {values['paired_task_count']}; analytic-residual wins: {values['primary_win_count']}/{values['paired_task_count']}.",
            f"Paired equal-resource NMSE delta (analytic residual minus Adam): mean {interval['mean']:.8g}, median {interval['median']:.8g}, deterministic percentile-bootstrap 95% CI [{interval['bootstrap_95_ci_low']:.8g}, {interval['bootstrap_95_ci_high']:.8g}].",
            "",
        ])
    (output_dir / "STATISTICS.md").write_text("\n".join(lines), encoding="utf-8")
    records = _load_results(run_dir)
    for split in sorted({record["split"] for record in records}):
        _svg_curve(output_dir / "figures" / f"learning_curves_{split}.svg", split, [record for record in records if record["split"] == split])
    return payload
