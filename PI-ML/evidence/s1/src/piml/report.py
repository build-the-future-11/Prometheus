"""Generate evidence-bound LaTeX fragments and dependency-free SVG figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


def _fmt(value: float) -> str:
    if abs(value) < 0.001 or abs(value) >= 10000:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _bar_svg(labels: list[str], values: list[float], title: str, ylabel: str) -> str:
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 60, 110
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(values) * 1.15 if max(values) > 0 else 1.0
    bar_width = plot_width / len(values) * 0.62
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="black"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        center = left + (index + 0.5) * plot_width / len(values)
        bar_height = value / maximum * plot_height
        y = top + plot_height - bar_height
        chunks.append(f'<rect x="{center-bar_width/2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2864b7"/>')
        chunks.append(f'<text x="{center:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="15">{escape(_fmt(value))}</text>')
        chunks.append(f'<text x="{center:.1f}" y="{top+plot_height+24}" transform="rotate(25 {center:.1f} {top+plot_height+24})" text-anchor="start" font-family="sans-serif" font-size="14">{escape(label)}</text>')
    chunks.append(f'<text x="20" y="{top+plot_height/2}" transform="rotate(-90 20 {top+plot_height/2})" text-anchor="middle" font-family="sans-serif" font-size="16">{escape(ylabel)}</text>')
    chunks.append('</svg>')
    return "\n".join(chunks) + "\n"


def generate(run_dir: Path, paper_dir: Path) -> dict[str, str]:
    analysis = json.loads((run_dir / "analysis.json").read_text())
    verification = json.loads((run_dir / "verification.json").read_text())
    receipt = json.loads((run_dir / "run_receipt.json").read_text())
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "figures").mkdir(exist_ok=True)
    summary = analysis["summaries"]
    iid = [entry for entry in summary if entry["split"] == "iid_test"]
    labels = [entry["condition"].replace("_", " ") for entry in iid]
    iterations = [float(entry["iterations_mean"]) for entry in iid]
    errors = [float(entry["relative_l2_mean"]) for entry in iid]
    (paper_dir / "figures" / "iid_iterations.svg").write_text(
        _bar_svg(labels, iterations, "IID mean Krylov iterations", "iterations")
    )
    (paper_dir / "figures" / "iid_error.svg").write_text(
        _bar_svg(labels, errors, "IID mean relative solution error", "relative L2 error")
    )
    paired = analysis["paired_primary"]
    decision = analysis["decision"]
    macros = [
        f"\\newcommand{{\\StudyDisposition}}{{{decision['disposition']}}}",
        f"\\newcommand{{\\RunPhase}}{{{receipt['phase']}}}",
        f"\\newcommand{{\\IIDReduction}}{{{100*float(paired['iid_test']['mean']):.1f}\\%}}",
        f"\\newcommand{{\\IIDReductionLow}}{{{100*float(paired['iid_test']['ci95_lower']):.1f}\\%}}",
        f"\\newcommand{{\\IIDReductionHigh}}{{{100*float(paired['iid_test']['ci95_upper']):.1f}\\%}}",
        f"\\newcommand{{\\OODReduction}}{{{100*float(paired['ood_test']['mean']):.1f}\\%}}",
        f"\\newcommand{{\\VerificationStatus}}{{{verification['status']}}}",
        f"\\newcommand{{\\ConfigDigest}}{{\\texttt{{{receipt['config_sha256'][:12]}}}}}",
    ]
    (paper_dir / "generated_results.tex").write_text("\n".join(macros) + "\n")
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Split & Condition & Rel. $L_2$ & Rel. residual & Iterations & Time ($\\mu$s) \\\\",
        "\\midrule",
    ]
    for entry in summary:
        lines.append(
            f"{entry['split'].replace('_', r'\_')} & {entry['condition'].replace('_', r'\_')} & "
            f"{_fmt(float(entry['relative_l2_mean']))} & {_fmt(float(entry['relative_residual_l2_mean']))} & "
            f"{_fmt(float(entry['iterations_mean']))} & {_fmt(float(entry['elapsed_ns_mean'])/1000)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (paper_dir / "results_table.tex").write_text("\n".join(lines) + "\n")
    facts = {
        "run_dir": str(run_dir),
        "receipt": receipt,
        "verification": verification,
        "decision": decision,
        "paired_primary": paired,
    }
    (paper_dir / "evidence_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    return {"disposition": str(decision["disposition"]), "verification": str(verification["status"])}

