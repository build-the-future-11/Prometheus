#!/usr/bin/env python3
"""Create provenance-bound descriptive analyses from a retained NEUROGENESIS bundle.

The script is deliberately read-only with respect to the input bundle.  It does
not run a method, select a configuration, or admit protected evaluation.  All
reported quantities are recomputed from retained result and trace artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable


ANALYSIS_SCHEMA = "neurogenesis.descriptive-analysis.v1"
WINDOW = 24


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def _stdev(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _fmt(value: float, digits: int = 4) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    index = (len(sorted_values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def _bootstrap_mean_ci(values: list[float], seed_text: str, draws: int = 10_000) -> dict[str, float | int]:
    """Return a deterministic descriptive percentile bootstrap interval.

    This interval is explicitly descriptive: five development identities are
    not a basis for a population-level efficacy claim.
    """

    if not values:
        return {"draws": 0, "lower": math.nan, "upper": math.nan}
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(_mean(values[rng.randrange(len(values))] for _ in values))
    means.sort()
    return {"draws": draws, "lower": _percentile(means, 0.025), "upper": _percentile(means, 0.975)}


def _rolling_mean(values: list[float], width: int = WINDOW) -> list[float]:
    total = 0.0
    result: list[float] = []
    for index, value in enumerate(values):
        total += value
        if index >= width:
            total -= values[index - width]
        result.append(total / min(index + 1, width))
    return result


def _read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_bundle(bundle: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    receipt_path = bundle / "receipt.json"
    manifest_path = bundle / "run_manifest.json"
    aggregate_path = bundle / "aggregate" / "summary.json"
    for path in (receipt_path, manifest_path, aggregate_path):
        if not path.is_file():
            raise ValueError(f"missing required bundle artifact: {path}")

    artifacts: dict[str, str] = {
        path.relative_to(bundle).as_posix(): _sha256(path)
        for path in (receipt_path, manifest_path, aggregate_path)
    }
    attempts: list[dict[str, Any]] = []
    for result_path in sorted((bundle / "attempts").glob("*/result.json")):
        result = _load_json(result_path)
        if result["status"] != "SUCCEEDED":
            attempts.append({"result": result, "trace": None, "attempt_dir": result_path.parent})
            artifacts[result_path.relative_to(bundle).as_posix()] = _sha256(result_path)
            continue
        trace_path = result_path.parent / "trace.jsonl"
        if not trace_path.is_file():
            raise ValueError(f"successful attempt has no trace: {result_path.parent}")
        trace = _read_trace(trace_path)
        if len(trace) != result["metrics"]["resource_usage"]["steps"]:
            raise ValueError(f"trace/result step count mismatch: {result_path.parent}")
        if hashlib.sha256(trace_path.read_bytes()).hexdigest() != result["trace_sha256"]:
            raise ValueError(f"trace hash mismatch: {result_path.parent}")
        attempts.append({"result": result, "trace": trace, "attempt_dir": result_path.parent})
        artifacts[result_path.relative_to(bundle).as_posix()] = _sha256(result_path)
        artifacts[trace_path.relative_to(bundle).as_posix()] = _sha256(trace_path)
    if not attempts:
        raise ValueError(f"no attempt results found under {bundle}")
    return attempts, artifacts


def build_analysis(bundle: Path) -> dict[str, Any]:
    attempts, artifact_hashes = load_bundle(bundle)
    succeeded = [entry for entry in attempts if entry["result"]["status"] == "SUCCEEDED"]
    policies = sorted({entry["result"]["policy_name"] for entry in attempts})
    by_policy: dict[str, list[dict[str, Any]]] = {policy: [] for policy in policies}
    for entry in attempts:
        by_policy[entry["result"]["policy_name"]].append(entry)

    policy_summaries: dict[str, Any] = {}
    curves: dict[str, list[float]] = {}
    for policy, entries in by_policy.items():
        completed = [entry for entry in entries if entry["trace"] is not None]
        result_metrics = [entry["result"]["metrics"] for entry in completed]
        rolling = [_rolling_mean([row["squared_error"] for row in entry["trace"]]) for entry in completed]
        if rolling:
            if len({len(curve) for curve in rolling}) != 1:
                raise ValueError(f"trace lengths differ within policy {policy}")
            curves[policy] = [_mean(curve[step] for curve in rolling) for step in range(len(rolling[0]))]
        else:
            curves[policy] = []
        policy_summaries[policy] = {
            "attempted": len(entries),
            "succeeded": len(completed),
            "failed": len(entries) - len(completed),
            "primary_mse_mean": _mean(metric["primary"]["value"] for metric in result_metrics),
            "primary_mse_sample_stdev": _stdev(metric["primary"]["value"] for metric in result_metrics),
            "adaptation_switch_mse_mean": _mean(metric["adaptation_switch_mse"] for metric in result_metrics),
            "retention_return_mse_mean": _mean(metric["retention_return_mse"] for metric in result_metrics),
            "wall_seconds_mean": _mean(metric["resource_usage"]["wall_seconds"] for metric in result_metrics),
            "peak_state_bytes_max": max((metric["resource_usage"]["peak_state_bytes"] for metric in result_metrics), default=None),
            "parameter_updates_total": sum(metric["resource_usage"]["parameter_updates"] for metric in result_metrics),
            "gradient_evaluations_total": sum(metric["resource_usage"]["gradient_evaluations"] for metric in result_metrics),
            "retrievals_total": sum(metric["resource_usage"]["retrievals"] for metric in result_metrics),
            "writes_total": sum(metric["resource_usage"]["writes"] for metric in result_metrics),
            "consolidations_total": sum(metric["resource_usage"]["consolidations"] for metric in result_metrics),
            "failed_attempt_ids": [entry["result"]["attempt_id"] for entry in entries if entry["trace"] is None],
        }

    frozen_by_seed = {
        entry["result"]["seed"]: entry["result"]["metrics"]["primary"]["value"]
        for entry in by_policy.get("no_adaptation", [])
        if entry["trace"] is not None
    }
    paired: dict[str, Any] = {}
    for policy, entries in by_policy.items():
        if policy == "no_adaptation":
            continue
        differences = [
            entry["result"]["metrics"]["primary"]["value"] - frozen_by_seed[entry["result"]["seed"]]
            for entry in entries
            if entry["trace"] is not None and entry["result"]["seed"] in frozen_by_seed
        ]
        paired[policy] = {
            "comparison": "policy_mse_minus_no_adaptation_mse",
            "paired_seed_count": len(differences),
            "paired_differences": differences,
            "mean_difference": _mean(differences),
            "sample_stdev": _stdev(differences),
            "descriptive_bootstrap_mean_ci_95": _bootstrap_mean_ci(differences, f"{bundle}:{policy}"),
            "negative_difference_count": sum(value < 0 for value in differences),
            "positive_difference_count": sum(value > 0 for value in differences),
            "zero_difference_count": sum(value == 0 for value in differences),
        }

    manifest = _load_json(bundle / "run_manifest.json")
    receipt = _load_json(bundle / "receipt.json")
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "claim_boundary": "Descriptive analysis of a development-only conventional-baseline bundle; no candidate mechanism or protected evaluation was run.",
        "bundle": bundle.as_posix(),
        "input_artifact_sha256": dict(sorted(artifact_hashes.items())),
        "input_artifact_map_sha256": hashlib.sha256(
            json.dumps(dict(sorted(artifact_hashes.items())), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "run_manifest": {
            "matrix_id": manifest["matrix_id"],
            "protocol_sha256": manifest["protocol_sha256"],
            "config_sha256": manifest["config_sha256"],
            "source_sha256": manifest["source_sha256"],
            "attempt_count": manifest["attempt_count"],
        },
        "run_accounting": {
            "requested": manifest["attempt_count"],
            "started": receipt["attempted"],
            "completed": receipt["succeeded"],
            "failed": receipt["failed"],
            "terminated": 0,
            "resumed": 0,
            "excluded": 0,
            "protected_evaluation_runs": 0,
        },
        "policy_summaries": policy_summaries,
        "paired_against_no_adaptation": paired,
        "curve_window": WINDOW,
        "rolling_mse_curves": curves,
        "regime_boundaries": [120, 240, 360],
    }


def _svg_document(width: int, height: int, body: str, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title><desc id="desc">Generated from a verified NEUROGENESIS development bundle.</desc>
<style>text{{font-family:Arial,sans-serif;fill:#172033}}.small{{font-size:12px}}.label{{font-size:14px}}.title{{font-size:20px;font-weight:700}}.axis{{stroke:#4d5b70;stroke-width:1}}.grid{{stroke:#d6dce6;stroke-width:1}}.regime{{stroke:#9aa7b8;stroke-width:1;stroke-dasharray:4 4}}</style>{body}</svg>'''


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{joined}"/>'


def render_learning_curves(analysis: dict[str, Any], output: Path) -> None:
    curves = analysis["rolling_mse_curves"]
    colors = {"no_adaptation": "#475569", "sgd_lr_0p01": "#0f766e", "sgd_lr_0p05": "#c2410c", "mechanism_off_adam_lr_0p01": "#7c3aed"}
    width, height = 1120, 590
    left, top, plot_width, plot_height = 88, 90, 930, 390
    nonempty = [point for curve in curves.values() for point in curve]
    ymax = max(nonempty) if nonempty else 1.0
    ymax = max(1.0, math.ceil(ymax * 10.0) / 10.0)
    body = [f'<text class="title" x="{left}" y="35">Development rolling prequential MSE (window = {analysis["curve_window"]})</text>',
            f'<text class="small" x="{left}" y="57">Five fixed development identities; descriptive only; lower is better.</text>',
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#4d5b70"/>']
    for fraction in range(6):
        y = top + plot_height - fraction * plot_height / 5
        value = ymax * fraction / 5
        body.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/><text class="small" x="18" y="{y + 4:.2f}">{value:.2f}</text>')
    for boundary in analysis["regime_boundaries"]:
        x = left + boundary / 479 * plot_width
        body.append(f'<line class="regime" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/><text class="small" x="{x + 4:.2f}" y="{top + 16}">switch {boundary}</text>')
    for policy, curve in curves.items():
        if not curve:
            continue
        points = [(left + step / (len(curve) - 1) * plot_width, top + plot_height - min(value, ymax) / ymax * plot_height) for step, value in enumerate(curve)]
        body.append(_polyline(points, colors.get(policy, "#111827")))
    legend_x = left
    for index, policy in enumerate(sorted(curves)):
        x = legend_x + index * 225
        body.append(f'<line x1="{x}" y1="520" x2="{x + 26}" y2="520" stroke="{colors.get(policy, "#111827")}" stroke-width="3"/><text class="small" x="{x + 32}" y="524">{html.escape(policy)}</text>')
    body.append(f'<text class="label" x="{left + plot_width / 2 - 64:.2f}" y="{top + plot_height + 43}">prediction step</text>')
    output.write_text(_svg_document(width, height, "".join(body), "Development rolling prequential MSE"), encoding="utf-8")


def render_tradeoff(analysis: dict[str, Any], output: Path) -> None:
    summaries = analysis["policy_summaries"]
    colors = {"no_adaptation": "#475569", "sgd_lr_0p01": "#0f766e", "sgd_lr_0p05": "#c2410c", "mechanism_off_adam_lr_0p01": "#7c3aed"}
    width, height = 990, 570
    left, top, plot_width, plot_height = 88, 90, 790, 360
    max_x = max(summary["adaptation_switch_mse_mean"] for summary in summaries.values()) * 1.08
    max_y = max(summary["retention_return_mse_mean"] for summary in summaries.values()) * 1.08
    body = [f'<text class="title" x="{left}" y="35">Stability--plasticity tradeoff on development streams</text>',
            f'<text class="small" x="{left}" y="57">x: first-24 post-switch MSE; y: first-24 return-regime MSE; lower-left is preferable.</text>',
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#4d5b70"/>']
    for fraction in range(6):
        x = left + fraction * plot_width / 5
        y = top + plot_height - fraction * plot_height / 5
        body.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/><text class="small" x="{x - 12:.2f}" y="{top + plot_height + 18}">{max_x * fraction / 5:.1f}</text>')
        body.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/><text class="small" x="20" y="{y + 4:.2f}">{max_y * fraction / 5:.1f}</text>')
    for policy, summary in summaries.items():
        x = left + summary["adaptation_switch_mse_mean"] / max_x * plot_width
        y = top + plot_height - summary["retention_return_mse_mean"] / max_y * plot_height
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{colors.get(policy, "#111827")}"/><text class="small" x="{x + 10:.2f}" y="{y - 9:.2f}">{html.escape(policy)}</text>')
    body.append(f'<text class="label" x="{left + 230}" y="{top + plot_height + 48}">mean post-switch MSE</text><text class="label" x="{left}" y="{top + plot_height + 73}">All results are development-only conventional controls.</text>')
    output.write_text(_svg_document(width, height, "".join(body), "Stability plasticity tradeoff"), encoding="utf-8")


def write_outputs(analysis: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {output}")
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    (output / "tables").mkdir()
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summaries = analysis["policy_summaries"]
    with (output / "tables" / "development_baselines.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["policy", "attempted", "succeeded", "failed", "primary_mse_mean", "primary_mse_sample_stdev", "adaptation_switch_mse_mean", "retention_return_mse_mean", "wall_seconds_mean", "peak_state_bytes_max", "parameter_updates_total", "gradient_evaluations_total", "retrievals_total", "writes_total", "consolidations_total"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for policy, summary in sorted(summaries.items()):
            writer.writerow({"policy": policy, **{name: summary[name] for name in columns[1:]}})
    latex_newline = r"\\"
    lines = ["\\begin{tabular}{lrrrrr}", "\\hline", f"Policy & Overall MSE & Switch MSE & Return MSE & State bytes & Updates {latex_newline}", "\\hline"]
    for policy, summary in sorted(summaries.items()):
        escaped_policy = policy.replace("_", r"\_")
        lines.append(f"{escaped_policy} & {_fmt(summary['primary_mse_mean'])} $\\pm$ {_fmt(summary['primary_mse_sample_stdev'])} & {_fmt(summary['adaptation_switch_mse_mean'])} & {_fmt(summary['retention_return_mse_mean'])} & {summary['peak_state_bytes_max']} & {summary['parameter_updates_total']} {latex_newline}")
    lines.extend(["\\hline", "\\end{tabular}", "% Development-only conventional baselines; generated from retained artifacts."])
    (output / "tables" / "development_baselines.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_learning_curves(analysis, output / "figures" / "development_rolling_mse.svg")
    render_tradeoff(analysis, output / "figures" / "stability_plasticity.svg")
    rows = "".join(f"<tr><td>{html.escape(policy)}</td><td>{_fmt(summary['primary_mse_mean'])} +/- {_fmt(summary['primary_mse_sample_stdev'])}</td><td>{_fmt(summary['adaptation_switch_mse_mean'])}</td><td>{_fmt(summary['retention_return_mse_mean'])}</td><td>{summary['parameter_updates_total']}</td></tr>" for policy, summary in sorted(summaries.items()))
    report = f"""<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>NEUROGENESIS development report</title><style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;color:#172033}}table{{border-collapse:collapse}}td,th{{border:1px solid #bac4d3;padding:.45rem;text-align:right}}td:first-child,th:first-child{{text-align:left}}.notice{{background:#fff5d6;padding:1rem;border-left:4px solid #bf7b00}}</style><h1>NEUROGENESIS development baseline report</h1><p class=\"notice\">{html.escape(analysis['claim_boundary'])}</p><p>Input artifact map SHA-256: <code>{analysis['input_artifact_map_sha256']}</code></p><table><thead><tr><th>Policy</th><th>Overall MSE, mean +/- SD</th><th>Switch MSE</th><th>Return MSE</th><th>Total updates</th></tr></thead><tbody>{rows}</tbody></table><h2>Figures</h2><img alt=\"rolling MSE curves\" src=\"figures/development_rolling_mse.svg\" width=\"100%\"><img alt=\"stability plasticity tradeoff\" src=\"figures/stability_plasticity.svg\" width=\"100%\"><h2>Provenance</h2><p>See <code>analysis.json</code> for every input artifact hash, raw paired differences, descriptive bootstrap intervals, and run accounting.</p></html>"""
    (output / "report.html").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = build_analysis(args.bundle)
    write_outputs(analysis, args.output)
    print(json.dumps({"status": "CREATED", "output": args.output.as_posix(), "input_artifact_map_sha256": analysis["input_artifact_map_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
