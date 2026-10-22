"""Generate paper numbers, tables, plots and an evidence map from verified runs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import mean

from scripts.verify_evidence import verify
from scripts.verify_resampling import verify_resampling
from scripts.compare_reproduction import compare


def tex(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def table(path, headers, rows, caption):
    columns = "l" + "r" * (len(headers)-1)
    body = [r"\begin{table}[htbp]\centering\small", r"\caption{" + caption + "}",
            r"\begin{tabular}{" + columns + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    body += [" & ".join(tex(x) for x in row) + r" \\" for row in rows]
    body += [r"\bottomrule", r"\end{tabular}\end{table}"]
    path.write_text("\n".join(body) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--robust-run", default="kaggle-robust-aggregation-v3-r1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "paper/generated"
    output.mkdir(parents=True, exist_ok=True)
    run_ids = ["kaggle-development-v1-r1", "kaggle-quantile-composition-v2-r1", args.robust_run]
    audits = [verify(root, run) for run in run_ids]
    resampling_audit = verify_resampling(root, args.robust_run)
    reproduction_audit = None
    reproduction_run = "kaggle-robust-aggregation-v3-reproduction-r1"
    if args.robust_run == "kaggle-robust-aggregation-v3-r1" and (root / "runs" / reproduction_run).exists():
        reproduction_audit = compare(root, args.robust_run, reproduction_run)
        if reproduction_audit["status"] != "EXACT_SCIENTIFIC_REPRODUCTION":
            raise ValueError("reproduction comparison failed; refuse paper success claims")
        table(output / "reproduction.tex", ["Execution", "Cells", "Seconds", "Peak RSS (MiB)"],
              [(label, 20, f"{reproduction_audit['runtime_seconds'][i]:.1f}", f"{reproduction_audit['peak_rss_mib'][i]:.1f}")
               for i, label in enumerate(["Original v3", "Fresh reproduction"])],
              "Identical frozen-source executions. Predictions, split files and all 20 checkpoints match byte for byte; both runs record zero pageout increase. Timing differences are environmental, not a model speedup.")
    runs = [root / "runs" / run for run in run_ids]
    old, quantile, robust = [json.loads((run / "results.json").read_text()) for run in runs]
    receipt = json.loads((runs[2] / "receipt.json").read_text())
    v1 = old["official_cv"]
    names = {"et_mean": "ExtraTrees mean", "et_median": "ExtraTrees median", "et_median_96": "Median, 96 trees",
             "et_fractions_median": "Median, fractions only", "random_forest": "Random forest", "hist_gradient_boosting": "Histogram boosting",
             "ridge": "Ridge", "train_median": "Training median"}
    by_model = {name: [m for m in robust["metrics"] if m["model"] == name] for name in names}
    macros = {"VOneMAE": f"{v1['metrics']['proposed']['mae_ev']:.3f}",
              "VOneRidge": f"{v1['metrics']['ridge']['mae_ev']:.3f}",
              "VTwoMAE": f"{quantile['metrics']['quantile_point']['mae_ev']:.3f}",
              "VThreeMedian": f"{mean(m['mae_ev'] for m in by_model['et_median']):.3f}",
              "VThreeMean": f"{mean(m['mae_ev'] for m in by_model['et_mean']):.3f}",
              "VThreeRidge": f"{mean(m['mae_ev'] for m in by_model['ridge']):.3f}",
              "VThreeDisposition": robust["disposition"],
              "VThreeDelta": f"{robust['primary']['mean_difference_ev']:.3f}",
              "VThreeLow": f"{robust['primary']['percentile_95_interval_ev'][0]:.3f}",
              "VThreeHigh": f"{robust['primary']['percentile_95_interval_ev'][1]:.3f}",
              "VThreeSeconds": f"{receipt['duration_seconds']:.1f}",
              "VThreeRSS": f"{receipt['resources']['peak_observed_rss_mib']:.1f}",
              "VThreeMedianRMSE": f"{mean(m['rmse_ev'] for m in by_model['et_median']):.3f}",
              "VThreeMeanRMSE": f"{mean(m['rmse_ev'] for m in by_model['et_mean']):.3f}",
              "VThreeUnits": str(robust["canonical_units"])}
    (output / "numbers.tex").write_text("% Generated; do not edit numerical values.\n" + "\n".join(r"\newcommand{" + chr(92) + key + "}{" + value + "}" for key, value in macros.items()) + "\n")
    table(output / "v1.tex", ["Model", "MAE (eV)", "RMSE (eV)"],
          [(name, f"{v1['metrics'][key]['mae_ev']:.3f}", f"{v1['metrics'][key]['rmse_ev']:.3f}") for key, name in [("proposed", "ExtraTrees mean"), ("ridge", "Ridge"), ("trivial", "Training median")]],
          "Historical official-split results (v1); canonical duplicates are retained.")
    table(output / "v3.tex", ["Model", "Repeat 1", "Repeat 2", "Repeat 3", "Mean MAE"],
          [(names[name], *[f"{m['mae_ev']:.3f}" for m in metrics], f"{mean(m['mae_ev'] for m in metrics):.3f}") for name, metrics in by_model.items()],
          "Grouped development MAE in eV. Repetitions share the same canonical units and are not independent datasets.")
    c90 = [c for c in robust["calibration"] if c["nominal_coverage"] == .9]
    methods = sorted({c["method"] for c in c90})
    table(output / "calibration.tex", ["Method", "Coverage", "Width (eV)", "Interval score"],
          [(name.replace("_", " "), *[f"{mean(c[k] for c in c90 if c['method']==name):.3f}" for k in ["observed_coverage", "mean_interval_width_ev", "mean_interval_score_ev"]]) for name in methods],
          "Mean across three repeats at nominal 90\\% coverage; normalized and constant-width residual calibration use the same proper-training models.")
    table(output / "holdouts.tex", ["Element", "Aggregation", "MAE (eV)", "Coverage", "Width (eV)"],
          [(h["element"], h["model"].replace("et_", ""), f"{h['mae_ev']:.3f}", f"{h['observed_coverage']:.3f}", f"{h['mean_interval_width_ev']:.3f}") for h in robust["holdouts"]],
          "Element-holdout development results. Intervals target nominal 90\\% coverage; exchangeability is not assumed under these shifts.")
    policies = sorted({r["policy"] for r in robust["ranking"]})
    table(output / "ranking.tex", ["Policy", "Mean hits / 50", "Precision", "Enrichment"],
          [(policy.replace("_", " "), *[f"{mean(r[k] for r in robust['ranking'] if r['policy']==policy and r['budget']==50):.3f}" for k in ["hits", "precision", "enrichment"]]) for policy in policies],
          "Retrospective candidate selection at budget 50, averaged across repetitions. Probability-like scores are working heuristics.")
    table(output / "probes.tex", ["Probe", "Fit units", "MAE (eV)", "RMSE (eV)"],
          [(p["probe"].replace("_", " "), p["fit_n"], f"{p['mae_ev']:.3f}", f"{p['rmse_ev']:.3f}") for p in robust["probes"]],
          "First-repeat, first-fold exploratory probes with 96 trees. These do not establish full scaling laws or general robustness.")
    diagnostic = json.loads((runs[2] / "quantile_diagnostic.json").read_text())
    table(output / "diagnostic.tex", ["Initialization", "First-tree nodes", "Distinct outputs", "Fixture MAE"],
          [(name.replace("_", " "), d["first_tree_nodes"], d["unique_final_predictions"], f"{d['training_mae_ev']:.4f}") for name, d in diagnostic["models"].items()],
          "Synthetic zero-mass diagnostic; training errors on a hand-defined fixture are not benchmark performance.")
    with (runs[2] / "tables/subgroups.csv").open() as stream:
        groups = list(csv.DictReader(stream))
    table(output / "subgroups.tex", ["Stratum", "Units", "Mean aggregation MAE", "Median aggregation MAE"],
          [(g, next(r["n"] for r in groups if r["subgroup"]==g), *[f"{mean(float(r['mae_ev']) for r in groups if r['subgroup']==g and r['model']==model):.3f}" for model in ["et_mean", "et_median"]]) for g in ["zero", "positive", "contains_oxygen", "conflicting_duplicates"]],
          "Subgroup absolute error in eV, averaged across repeats. Subgroups overlap; the three conflicting duplicate units are too few for stable inference.")
    with (runs[2] / "tables/costs.csv").open() as stream:
        costs = list(csv.DictReader(stream))
    table(output / "costs.tex", ["Model operation", "Cells", "Mean seconds", "Max seconds"],
          [(name.replace("_", " "), len([r for r in costs if r["model"]==name]),
            f"{mean(float(r['fit_predict_seconds']) for r in costs if r['model']==name):.3f}",
            f"{max(float(r['fit_predict_seconds']) for r in costs if r['model']==name):.3f}") for name in sorted({r["model"] for r in costs})],
          "Measured outer-cell fit/prediction costs. Shared ExtraTrees timing includes calibration predictions; checkpoint I/O is outside that timer. Node counts are retained in the raw cost table.")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    labels = list(names)
    axes[0].barh([names[k] for k in labels], [mean(m["mae_ev"] for m in by_model[k]) for k in labels], color=["#287b8e" if k=="et_median" else "#6b8092" for k in labels])
    axes[0].invert_yaxis(); axes[0].set_xlabel("Grouped MAE (eV)")
    delta = robust["primary"]["median_minus_mean_mae_ev_by_repeat"]
    axes[1].bar([1,2,3], delta, color="#287b8e")
    axes[1].axhline(-.01, color="#bb6644", linestyle="--", label="Required -0.01 eV")
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set(xticks=[1,2,3], xlabel="Grouped split repetition", ylabel="Median minus mean MAE (eV)")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "aggregation.pdf"); fig.savefig(output / "aggregation.png"); plt.close(fig)
    fig, axes = plt.subplots(1,2, figsize=(9,3.4), layout="constrained")
    for method in ["mean_normalized", "median_normalized", "median_constant"]:
        coverages = [.5,.8,.9,.95]
        observed = [mean(c["observed_coverage"] for c in robust["calibration"] if c["method"]==method and c["nominal_coverage"]==q) for q in coverages]
        axes[0].plot(coverages, observed, "o-", label=method.replace("_", " "))
    axes[0].plot([.45,1],[.45,1], "k--", linewidth=.7)
    axes[0].set(xlabel="Nominal coverage", ylabel="Observed coverage")
    axes[0].legend(fontsize=7)
    for method in ["et_mean", "et_median"]:
        h = [h for h in robust["holdouts"] if h["model"]==method]
        axes[1].plot([h["element"] for h in h], [h["observed_coverage"] for h in h], "o-", label=method)
    axes[1].axhline(.9, color="black", linestyle="--", linewidth=.7)
    axes[1].set(xlabel="Held-out element", ylabel="Observed 90% coverage", ylim=(0,1))
    axes[1].legend(fontsize=8)
    fig.savefig(output / "calibration_shift.pdf"); fig.savefig(output / "calibration_shift.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4,3.2), layout="constrained")
    for policy in policies:
        budgets = [10,25,50,100]
        precision = [mean(r["precision"] for r in robust["ranking"] if r["policy"]==policy and r["budget"]==b) for b in budgets]
        ax.plot(budgets, precision, "o-", label=policy.replace("_", " "))
    ax.set(xlabel="Selection budget", ylabel="Retrospective precision", ylim=(0,1))
    ax.legend(fontsize=8, ncols=2)
    fig.savefig(output / "selection.pdf"); fig.savefig(output / "selection.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,2.6), layout="constrained")
    ax.set_xlim(0,10); ax.set_ylim(0,3); ax.axis("off")
    boxes = [(0.8,1.5,"Pinned formulas\nand source labels"), (3.0,1.5,"Canonical groups\nFit / calibrate / test"),
             (5.2,1.5,"Same 288 trees\nMean vs median"), (7.4,1.5,"Intervals / ranking\nSaved predictions"), (9.6,1.5,"Independent\nnumeric audit")]
    ax.set_xlim(-0.3,10.7)
    for x,y,label in boxes:
        ax.text(x,y,label,ha="center",va="center",fontsize=9,bbox={"boxstyle":"round,pad=0.5","facecolor":"#e9f0f3","edgecolor":"#287b8e"})
    for a,b in zip(boxes,boxes[1:]):
        ax.annotate("", xy=(b[0]-.85,b[1]), xytext=(a[0]+.85,a[1]), arrowprops={"arrowstyle":"->","color":"#334b5d"})
    ax.text(5,.35,"Frozen development protocol  |  all outcomes retained  |  no protected external test",ha="center",fontsize=9)
    fig.savefig(output / "workflow.pdf"); fig.savefig(output / "workflow.png"); plt.close(fig)
    manifest = {"schema_version": 2, "runs": run_ids, "verification": audits, "resampling_verification": resampling_audit, "reproduction_verification": reproduction_audit,
                "claims": {"historical_point_performance": "runs/kaggle-development-v1-r1/predictions/oof_predictions.csv",
                           "aggregation_effect": f"runs/{args.robust_run}/predictions/oof.csv",
                           "retrospective_ranking": f"runs/{args.robust_run}/predictions/ranking.csv",
                           "shift_failures": f"runs/{args.robust_run}/predictions/holdouts.csv",
                           "quantile_mechanism": f"runs/{args.robust_run}/quantile_diagnostic.json"},
                "files": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file() and p.name != "evidence-map.json"}}
    (output / "evidence-map.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "GENERATED_FROM_VERIFIED_EVIDENCE", "output": str(output), "macros": macros}, indent=2))


if __name__ == "__main__":
    main()
