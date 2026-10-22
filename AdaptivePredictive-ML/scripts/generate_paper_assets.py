#!/usr/bin/env python3
"""Generate LaTeX tables and PGFPlots figures from retained v2 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts/kaggle-development-v2-r1/development-analysis-v2.json"
VERIFICATION = ROOT / "artifacts/kaggle-development-v2-r1/independent-verification-v2.json"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

METHOD_LABELS = {
    "adahedge": "AdaHedge",
    "hedge": "Hedge",
    "fixed_share": "Fixed share",
    "directed_fixed_share": "Directed fixed share",
    "adaptive_share": "Adaptive share",
    "ddfs": r"\textsc{DDFS}",
    "uniform": "Uniform ensemble",
    "seasonal_naive": "Seasonal naive",
}
CONTRAST_LABELS = {
    "ddfs_minus_fixed_share": r"\textsc{DDFS} $-$ fixed share",
    "ddfs_minus_adaptive_share": r"\textsc{DDFS} $-$ adaptive share",
    "ddfs_minus_directed_fixed_share": r"\textsc{DDFS} $-$ directed fixed",
    "directed_fixed_share_minus_fixed_share": "Directed fixed $-$ fixed share",
    "adaptive_share_minus_fixed_share": "Adaptive share $-$ fixed share",
    "ddfs_minus_adahedge": r"\textsc{DDFS} $-$ AdaHedge",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(ANALYSIS.read_text())
    verification = json.loads(VERIFICATION.read_text())
    if not verification.get("valid"):
        raise RuntimeError("refusing to generate paper assets from unverified evidence")
    if analysis["mean_series_weighted_mase"] != verification["mean_series_weighted_mase"]:
        # The verifier may differ at the final binary digit because it uses math.fsum.
        for method, value in analysis["mean_series_weighted_mase"].items():
            if abs(value - verification["mean_series_weighted_mase"][method]) > 1e-12:
                raise RuntimeError(f"analysis/verifier mismatch for {method}")

    ranked = sorted(analysis["mean_series_weighted_mase"].items(), key=lambda item: item[1])
    result_rows = [
        f"{rank} & {METHOD_LABELS[method]} & {value:.6f} \\\\"
        for rank, (method, value) in enumerate(ranked, start=1)
    ]
    write(
        TABLES / "results.tex",
        """\\begin{table}[H]
\\centering
\\caption{Equal-series mean MASE on the registered 2016 development window. Lower is better; every method has 3,660 timestamp-level predictions.}
\\label{tab:results}
\\begin{tabular}{r l S[table-format=1.6]}
\\toprule
Rank & Method & {Mean MASE} \\\\
\\midrule
""" + "\n".join(result_rows) + """
\\bottomrule
\\end{tabular}
\\end{table}""",
    )

    contrast_rows = []
    for name, label in CONTRAST_LABELS.items():
        item = analysis["contrasts"][name]
        boot = item["bootstrap"]
        contrast_rows.append(
            f"{label} & {boot['estimate']:.6f} & "
            f"[{boot['ci_lower']:.6f}, {boot['ci_upper']:.6f}] & "
            f"{item['series_wins']}/{item['series_count']} \\\\"
        )
    write(
        TABLES / "contrasts.tex",
        """\\begin{table}[h]
\\centering
\\small
\\caption{All preregistered development contrasts. Positive estimates mean the left-hand method has higher (worse) MASE. Intervals use 5,000 synchronized seven-day moving-block draws.}
\\label{tab:contrasts}
\\begin{tabular}{l r l r}
\\toprule
Contrast & Estimate & 95\\% interval & Left wins \\\\
\\midrule
""" + "\n".join(contrast_rows) + """
\\bottomrule
\\end{tabular}
\\end{table}""",
    )

    primary_series = analysis["contrasts"]["ddfs_minus_fixed_share"]["series_differences"]
    series_rows = [
        f"{series} & {value:.6f} & {'Favorable' if value < 0 else 'Unfavorable'} \\\\"
        for series, value in sorted(primary_series.items())
    ]
    write(
        TABLES / "series-effects.tex",
        """\\begin{table}[H]
\\centering
\\small
\\caption{Per-series primary development effects. Positive values indicate higher (worse) MASE for \\textsc{DDFS} than constant fixed share.}
\\label{tab:series-effects}
\\begin{tabular}{l r l}
\\toprule
Series & Difference & Direction \\\\
\\midrule
""" + "\n".join(series_rows) + """
\\bottomrule
\\end{tabular}
\\end{table}""",
    )

    plot_order = list(reversed(ranked))
    ylabels = ",".join(METHOD_LABELS[method] for method, _ in plot_order)
    yindex = {method: index for index, (method, _) in enumerate(plot_order)}
    ordinary = " ".join(
        f"({value:.9f},{yindex[method]})"
        for method, value in ranked if method != "ddfs"
    )
    ddfs_value = analysis["mean_series_weighted_mase"]["ddfs"]
    mase_figure = (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "  xbar,\n"
        "  width=0.84\\linewidth,\n"
        "  height=6.0cm,\n"
        "  xmin=0,\n"
        "  xmax=1.10,\n"
        "  xlabel={Mean MASE},\n"
        f"  ytick={{{','.join(str(index) for index in range(len(plot_order)))}}},\n"
        f"  yticklabels={{{ylabels}}},\n"
        "  yticklabel style={font=\\small},\n"
        "  xticklabel style={font=\\small},\n"
        "  bar width=5pt,\n"
        "  nodes near coords,\n"
        "  nodes near coords style={font=\\scriptsize,/pgf/number format/fixed,/pgf/number format/precision=3},\n"
        "  grid=major,\n"
        "  grid style={gray!20},\n"
        "  axis line style={black!60},\n"
        "]\n"
        f"\\addplot+[draw=blue!65!black,fill=blue!35] coordinates {{{ordinary}}};\n"
        f"\\addplot+[draw=red!70!black,fill=red!45] coordinates {{({ddfs_value:.9f},{yindex['ddfs']})}};\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}"
    )
    write(FIGURES / "mase-bars.tex", mase_figure)

    ordered_contrasts = list(CONTRAST_LABELS)
    contrast_plot_order = list(reversed(ordered_contrasts))
    contrast_ylabels = ",".join(CONTRAST_LABELS[name] for name in contrast_plot_order)
    contrast_yindex = {name: index for index, name in enumerate(contrast_plot_order)}
    interval_plots = []
    point_coordinates = []
    for name in ordered_contrasts:
        boot = analysis["contrasts"][name]["bootstrap"]
        interval_plots.append(
            "\\addplot[black,thick,mark=none] coordinates "
            f"{{({boot['ci_lower']:.9f},{contrast_yindex[name]}) ({boot['ci_upper']:.9f},{contrast_yindex[name]})}};"
        )
        point_coordinates.append(f"({boot['estimate']:.9f},{contrast_yindex[name]})")
    contrast_figure = (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "  width=0.76\\linewidth,\n"
        "  height=6.8cm,\n"
        "  xmin=-0.001,\n"
        "  xmax=0.017,\n"
        "  xlabel={Left-minus-right MASE (positive is worse)},\n"
        f"  ytick={{{','.join(str(index) for index in range(len(contrast_plot_order)))}}},\n"
        f"  yticklabels={{{contrast_ylabels}}},\n"
        "  yticklabel style={font=\\scriptsize},\n"
        "  xticklabel style={font=\\small},\n"
        "  grid=major,\n"
        "  grid style={gray!20},\n"
        "  axis line style={black!60},\n"
        "  extra x ticks={0},\n"
        "  extra x tick style={grid=major,grid style={black,dashed,thick}},\n"
        "]\n"
        + "\n".join(interval_plots)
        + "\n"
        + "\\addplot+[only marks,mark=*,mark size=2.3pt,red!75!black] coordinates {"
        + " ".join(point_coordinates)
        + "};\n"
        + "\\end{axis}\n"
        + "\\end{tikzpicture}"
    )
    write(FIGURES / "contrasts.tex", contrast_figure)

    generated = [
        TABLES / "results.tex",
        TABLES / "contrasts.tex",
        TABLES / "series-effects.tex",
        FIGURES / "mase-bars.tex",
        FIGURES / "contrasts.tex",
    ]
    manifest = {
        "schema_version": "adaptive-predictive.paper-assets.v1",
        "inputs": {
            str(ANALYSIS.relative_to(ROOT)): digest(ANALYSIS),
            str(VERIFICATION.relative_to(ROOT)): digest(VERIFICATION),
        },
        "outputs": {str(path.relative_to(ROOT)): digest(path) for path in generated},
        "claim_boundary": "Development-stage negative result; no protected confirmation, generalization, efficiency, or SOTA claim.",
    }
    write(PAPER / "asset-manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
