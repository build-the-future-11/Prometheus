"""Generate v4 publication artifacts only from verified retained evidence."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import read, sha, verify  # noqa: E402
from transfer import verify as verify_transfer  # noqa: E402


NAMES = {
    "candidate": "Memory gate",
    "baseline": "Persistent-ridge baseline",
    "all_update": "Always-update bank",
    "capacity_one": "Capacity 1",
    "uniform_bank": "Uniform bank",
    "no_bank": "No bank",
    "periodic": "Periodic solve",
    "replay": "FIFO replay",
    "knn": "FIFO kNN",
    "frozen": "Frozen zero",
    "last_observation": "Last observation",
}
COLORS = {
    "candidate": "#0b7189",
    "baseline": "#b5542d",
    "all_update": "#6b57a5",
    "periodic": "#26734d",
    "last_observation": "#777777",
}


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def save(fig, path: Path) -> None:
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def require_exact_receipt(run: Path) -> dict:
    receipt = read(run / "verification-exact.json")
    if receipt["status"] != "VALID" or not receipt["exact_replay"] or receipt["failed"]:
        raise ValueError(f"exact replay gate failed: {run}")
    return receipt


def generate(lock_path, analysis_path, confirmation, air, beijing, output, hosted_receipt=None) -> None:
    lock_path = Path(lock_path).resolve()
    analysis_path = Path(analysis_path).resolve()
    confirmation = Path(confirmation).resolve()
    air = Path(air).resolve()
    beijing = Path(beijing).resolve()
    output = Path(output)
    hosted = None
    if hosted_receipt is not None:
        hosted = read(Path(hosted_receipt).resolve())
        if (
            hosted.get("status") != "VALID"
            or hosted.get("platform") != "kaggle"
            or not hosted.get("exact_replay_valid")
            or not hosted.get("audits_valid")
        ):
            raise ValueError("hosted receipt gate failed")
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir()
    tables.mkdir()

    lock = read(lock_path)
    analysis = read(analysis_path)
    confirmation_check = verify(confirmation)
    confirmation_exact = require_exact_receipt(confirmation)
    confirmation_audit = read(confirmation / "audit.json")
    if (
        analysis["disposition"] not in {"SUPPORTED", "NEGATIVE", "INCONCLUSIVE"}
        or analysis["verification"]["artifact_map_sha256"]
        != confirmation_check["artifact_map_sha256"]
        or confirmation_exact["artifact_map_sha256"]
        != confirmation_check["artifact_map_sha256"]
        or confirmation_audit["status"] != "VALID"
        or confirmation_audit["failed"]
    ):
        raise ValueError("synthetic evidence gate failed")

    air_check = verify_transfer(air, lock_path)
    beijing_check = verify_transfer(beijing, lock_path)
    air_exact = require_exact_receipt(air)
    beijing_exact = require_exact_receipt(beijing)
    for name, check, exact in (
        ("air", air_check, air_exact),
        ("beijing", beijing_check, beijing_exact),
    ):
        audit = read((air if name == "air" else beijing) / "audit.json")
        if (
            check["status"] != "VALID"
            or check["failed"]
            or check["scientific_comparison"] != "COMPLETE"
            or exact["failed"]
            or audit["status"] != "VALID"
            or audit["failed"]
        ):
            raise ValueError(f"{name} evidence gate failed")

    air_summary = read(air / "summary.json")
    beijing_summary = read(beijing / "summary.json")
    rows = [read(path) for path in sorted((confirmation / "attempts").glob("*/result.json"))]
    methods = list(lock["configs"])
    conditions = lock["protocol"]["conditions"]
    primary = set(lock["protocol"]["primary_conditions"])
    primary_rows = [row for row in rows if row["task_id"].rsplit("-", 1)[0] in primary]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
        }
    )

    heat = np.array(
        [
            [
                analysis["per_condition_mse"][condition][method]
                / analysis["per_condition_mse"][condition]["baseline"]
                for condition in conditions
            ]
            for method in methods
        ]
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    image = ax.imshow(np.log2(heat), cmap="RdBu_r", vmin=-1, vmax=2, aspect="auto")
    ax.set_xticks(
        range(len(conditions)),
        ["Base", "Novel", "Delay 16", "Noise .2", "Distractors", "16 dims", "Horizon x2"],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(range(len(methods)), [NAMES[name] for name in methods])
    for i in range(len(methods)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Fresh-seed MSE relative to persistent-ridge baseline")
    fig.colorbar(image, ax=ax, label="log2 MSE ratio", shrink=0.8)
    save(fig, figures / "condition_heatmap")

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    for method in ("candidate", "baseline", "all_update", "periodic"):
        curves = []
        for path in sorted((confirmation / "attempts").glob(f"base-*__{method}/trace.jsonl.gz")):
            losses = np.array([event["squared_error"] for event in map(json.loads, gzip.open(path, "rt"))])
            cumulative = np.cumsum(np.r_[0, losses])
            curves.append(
                np.array(
                    [
                        (cumulative[i + 1] - cumulative[max(0, i - 23)])
                        / (i + 1 - max(0, i - 23))
                        for i in range(len(losses))
                    ]
                )
            )
        ax.plot(
            np.mean(curves, axis=0),
            label=NAMES[method],
            color=COLORS[method],
            linewidth=1.35,
        )
    for step in range(120, 960, 120):
        ax.axvline(step, color="#bbbbbb", linestyle=":", linewidth=0.7)
    ax.set(
        xlabel="Prediction step (zero-based)",
        ylabel="24-step mean squared error",
        title="All 20 held-out base-condition seeds",
    )
    ax.legend(ncol=2, fontsize=8)
    save(fig, figures / "learning_curves")

    table = []
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for method in methods:
        subset = [row for row in primary_rows if row["method"] == method]
        item = {
            "method": method,
            "mse": statistics.mean(row["mse"] for row in subset),
            "weight_solves": statistics.mean(row["usage"]["updates"] for row in subset),
            "information_updates": statistics.mean(
                row["usage"]["information_updates"] for row in subset
            ),
            "coordinate_ops": statistics.mean(row["usage"]["coordinate_ops"] for row in subset),
            "peak_state_bytes": max(row["peak_state_bytes"] for row in subset),
            "wall_seconds": statistics.mean(row["wall_seconds"] for row in subset),
            "attempts": len(subset),
        }
        table.append(item)
        for axis, key in zip(axes, ("weight_solves", "coordinate_ops")):
            axis.scatter(item[key], item["mse"], s=38, color=COLORS.get(method, "#7f8994"))
            axis.annotate(NAMES[method], (item[key], item["mse"]), xytext=(5, 4), textcoords="offset points", fontsize=6)
    axes[0].set(
        xlabel="Mean weight solves",
        ylabel="Mean primary-condition MSE",
        title="Solves are not total computation",
    )
    axes[1].set(
        xlabel="Counted scalar arithmetic",
        ylabel="Mean primary-condition MSE",
        title="Reference arithmetic / accuracy",
    )
    save(fig, figures / "efficiency")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=False)
    for axis, summary, title, ylabel in (
        (axes[0], air_summary, "Exposed Air Quality regression", "CO MSE (original units squared)"),
        (axes[1], beijing_summary, "Protected Beijing transfer", "PM2.5 MSE (original units squared)"),
    ):
        for method in ("candidate", "baseline", "periodic", "last_observation"):
            axis.plot(
                range(1, 8),
                summary["results"][method]["test_block_mse_original_units"],
                marker="o",
                markersize=3,
                label=NAMES[method],
                color=COLORS[method],
            )
        axis.set(xlabel="Contiguous evaluation block", ylabel=ylabel, title=title)
    axes[1].legend(fontsize=7, ncol=2)
    save(fig, figures / "real_transfer")

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.15, 1.7, 1.7, "Current features"),
        (2.2, 1.7, 2.4, "Persistent ridge + frozen bank"),
        (5.0, 1.7, 1.65, "Legal loss gate"),
        (7.1, 1.7, 1.6, "Prediction"),
        (6.5, 0.25, 3.2, "Delayed label + saved outputs"),
        (4.0, 0.25, 2.1, "Update EW credit"),
        (0.1, 0.25, 3.4, "Information update/solve or skip; FIFO archive"),
    ]
    for x, y, width, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                0.65,
                boxstyle="round,pad=.04",
                facecolor="#eaf3f5",
                edgecolor="#356573",
            )
        )
        ax.text(x + width / 2, y + 0.325, label, ha="center", va="center", fontsize=8)
    for start, end in [
        ((1.9, 2.025), (2.15, 2.025)),
        ((4.65, 2.025), (4.95, 2.025)),
        ((6.7, 2.025), (7.05, 2.025)),
        ((7.9, 1.65), (7.9, 0.95)),
        ((6.45, 0.575), (6.15, 0.575)),
        ((3.95, 0.575), (3.55, 0.575)),
        ((1.8, 0.95), (3.4, 1.65)),
    ]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#536979"})
    save(fig, figures / "architecture")

    with (tables / "primary_results.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (tables / "real_results.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "method", "mse_original_units", "scored", "weight_solves"])
        for dataset, summary in (("air", air_summary), ("beijing", beijing_summary)):
            for method, result in summary["results"].items():
                writer.writerow(
                    [dataset, method, result["test_mse_original_units"], result["test_scored"], result["test_updates"]]
                )

    primary_latex = [
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"Method & MSE & Solves & Info. updates & Arithmetic & State bytes\\\midrule",
    ]
    for row in table:
        primary_latex.append(
            f"{NAMES[row['method']]} & {row['mse']:.4f} & {row['weight_solves']:.1f} & "
            f"{row['information_updates']:.1f} & {row['coordinate_ops']:.0f} & {row['peak_state_bytes']} \\\\"
        )
    primary_latex.extend(
        [
            r"\bottomrule\end{tabular}",
            r"\caption{Primary-condition means over 100 task instances per method. State is numerical state, not process RSS.}",
            r"\label{tab:primary}\end{table}",
        ]
    )
    (tables / "primary_results.tex").write_text("\n".join(primary_latex) + "\n", encoding="utf-8")

    real_latex = [
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{llrrr}\toprule",
        r"Stream & Method & MSE & Scored & Solves\\\midrule",
    ]
    for dataset, summary in (("Air", air_summary), ("Beijing", beijing_summary)):
        for method, result in summary["results"].items():
            real_latex.append(
                f"{dataset} & {NAMES[method]} & {result['test_mse_original_units']:.6f} & "
                f"{result['test_scored']} & {result['test_updates']} \\\\"
            )
    real_latex.extend(
        [
            r"\bottomrule\end{tabular}",
            r"\caption{Complete chronological evaluation results in original target units squared.}",
            r"\label{tab:real}\end{table}",
        ]
    )
    (tables / "real_results.tex").write_text("\n".join(real_latex) + "\n", encoding="utf-8")

    primary_effect = analysis["primary"]
    all_update = analysis["comparisons"]["all_update"]
    periodic = analysis["comparisons"]["periodic"]
    beijing_effect = 100 * beijing_summary["relative_mse_change"]
    beijing_reduction = 100 * beijing_summary["update_reduction"]
    air_best = min(air_summary["results"], key=lambda name: air_summary["results"][name]["test_mse_original_units"])
    beijing_best = min(beijing_summary["results"], key=lambda name: beijing_summary["results"][name]["test_mse_original_units"])
    lines = [
        r"\subsection{Frozen synthetic confirmation}",
        f"The single registered campaign completed {len(rows):,} attempts with zero failed cells. "
        f"The predefined synthetic disposition is \\textbf{{{analysis['disposition']}}}. "
        f"The mean paired seed-level relative MSE change is {100 * primary_effect['relative_mse_change']:.2f}\\% "
        f"(95\\% bootstrap interval [{100 * primary_effect['ci95'][0]:.2f}, {100 * primary_effect['ci95'][1]:.2f}]\\%), "
        f"with {100 * primary_effect['mean_update_reduction']:.2f}\\% fewer weight solves.",
        f"Against the always-update bank, the relative MSE change is {100 * all_update['relative_mse_change']:.2f}\\% "
        f"(interval [{100 * all_update['ci95'][0]:.2f}, {100 * all_update['ci95'][1]:.2f}]\\%). "
        f"Against periodic solving it is {100 * periodic['relative_mse_change']:.2f}\\%, while the candidate uses "
        f"{-100 * periodic['mean_update_reduction']:.2f}\\% more solves than periodic. Solve reduction alone therefore does not explain the result.",
        r"\input{tables/primary_results.tex}",
        r"\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/condition_heatmap.pdf}\caption{Complete held-out condition matrix. Ratios are pooled condition means; the registered confidence interval uses paired seed-level effects.}\end{figure}",
        r"\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/learning_curves.pdf}\caption{Twenty-seed rolling errors for the base recurrence condition. Generator changes are hidden from all methods.}\end{figure}",
        r"\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/efficiency.pdf}\caption{Accuracy versus weight solves and counted reference arithmetic. Weight solves are not a hardware speed or energy measure.}\end{figure}",
        r"\subsection{Chronological streams}",
        f"All registered methods completed both chronological streams. On the exposed Air Quality regression stream, descriptive support is {str(air_summary['descriptive_support']).lower()}. "
        f"On the preselected Beijing PM2.5 stream, the candidate's relative MSE change versus the synthetic-selected baseline is {beijing_effect:.2f}\\%, "
        f"with {beijing_reduction:.2f}\\% fewer evaluation-period weight solves; descriptive support is {str(beijing_summary['descriptive_support']).lower()}. "
        f"The best registered method by MSE is {NAMES[air_best]} on Air and {NAMES[beijing_best]} on Beijing; "
        r"the candidate therefore is not the strongest overall baseline. These are single chronological streams with artificial four-hour label delay, not independent dataset replications or IID tests.",
        r"\input{tables/real_results.tex}",
        r"\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/real_transfer.pdf}\caption{Seven contiguous evaluation blocks. Blocks expose temporal variation but are not independent replicates.}\end{figure}",
    ]
    (output / "results.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    claims = {
        "synthetic_disposition": analysis["disposition"],
        "primary_relative_mse_change": primary_effect,
        "all_update_comparison": all_update,
        "periodic_comparison": periodic,
        "air_descriptive_support": air_summary["descriptive_support"],
        "beijing_descriptive_support": beijing_summary["descriptive_support"],
        "beijing_relative_mse_change": beijing_summary["relative_mse_change"],
        "beijing_update_reduction": beijing_summary["update_reduction"],
        "air_best_method": air_best,
        "beijing_best_method": beijing_best,
        "confirmation_verification": confirmation_exact,
        "air_verification": air_exact,
        "beijing_verification": beijing_exact,
        "independent_external_replication": False,
        "hosted_execution": hosted is not None,
        "hosted_receipt": hosted,
        "sota_demonstrated": False,
        "input_hashes": {
            str(path): sha(path)
            for path in (
                lock_path,
                analysis_path,
                confirmation / "manifest.json",
                air / "summary.json",
                beijing / "summary.json",
            )
        },
    }
    write_json(output / "claims.json", claims)
    write_json(
        output / "manifest.json",
        {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    )
    print(json.dumps({"status": "GENERATED", "claims": claims}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--air", type=Path, required=True)
    parser.add_argument("--beijing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hosted-receipt", type=Path)
    args = parser.parse_args()
    generate(
        args.lock,
        args.analysis,
        args.confirmation,
        args.air,
        args.beijing,
        args.output,
        args.hosted_receipt,
    )
