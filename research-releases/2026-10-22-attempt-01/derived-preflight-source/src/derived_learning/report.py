"""Generate Markdown tables exclusively from saved experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LABELS = {
    "adam": "Tuned Adam baseline",
    "analytic_residual": "Analytic residual targets (primary)",
    "block_jacobi_residual": "Simultaneous residual-target ablation",
    "block_gradient_no_derivation": "No-derivation block-gradient ablation",
    "fixed_normalized_lms": "Fixed normalized-LMS rule",
    "classical_backfitting": "Classical cyclic ridge backfitting",
    "global_ridge": "Global closed-form ridge baseline",
}


def _number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "not reached"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}g}"


def render_report(run_dir: Path) -> str:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# Generated experiment report",
        "",
        f"Study: `{summary['study_id']}`. Manifest status: `{manifest['status']}`. All numbers below are generated from `summary.json`; no table value is hand-entered.",
        "",
    ]
    for split, split_summary in summary["by_split"].items():
        lines.extend(
            [
                f"## {split.replace('_', ' ').title()}",
                "",
                "| Condition | Tasks ok/failed | Terminal mean NMSE | Equal-resource mean NMSE | Threshold reached | Median work to threshold | Mean terminal work |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        ordered_methods = [
            method
            for method in ("adam", "global_ridge", "analytic_residual", "classical_backfitting", "block_jacobi_residual", "block_gradient_no_derivation", "fixed_normalized_lms")
            if method in split_summary["methods"]
        ]
        ordered_methods.extend(sorted(set(split_summary["methods"]) - set(ordered_methods)))
        for method in ordered_methods:
            values = split_summary["methods"][method]
            lines.append(
                f"| {LABELS.get(method, method)} | {values['tasks_succeeded']}/{values['tasks_failed']} | "
                f"{_number(values['terminal_mean_nmse'])} | {_number(values['equal_resource_mean_nmse'])} | "
                f"{values['threshold_reached_count']} | {_number(values['median_work_units_to_threshold'])} | "
                f"{_number(values['mean_terminal_work_units'])} |"
            )
        criteria = split_summary["success_criteria"]
        lines.extend(
            [
                "",
                f"Paired tasks: {split_summary['paired_task_count']}; primary equal-resource wins: "
                f"{split_summary['paired_primary_win_count']}/{split_summary['paired_task_count']}. "
                f"Relative equal-resource improvement versus Adam: {_number(split_summary['relative_equal_resource_improvement_vs_adam'])}. ",
                "",
                f"Predeclared quality criterion: `{'PASS' if criteria['quality_pass'] else 'FAIL'}`. "
                f"Efficiency criterion: `{'PASS' if criteria['efficiency_pass'] else 'FAIL'}`. "
                f"Split interpretation: `{split_summary['interpretation']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Locked selection",
            "",
            "Final and long-horizon task answers were not used for selection. The locked parameters were:",
            "",
        ]
    )
    for method, parameters in selection["selected"].items():
        lines.append(f"- `{method}`: `{json.dumps(parameters, sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is an internally executed synthetic study, not independent reproduction. Passing would support only the frozen grouped-additive matrix under the recorded CPU accounting model; it would not establish a general learning mechanism. A failed criterion is retained as negative evidence and is not repaired by changing the endpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(run_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(run_dir), encoding="utf-8")

