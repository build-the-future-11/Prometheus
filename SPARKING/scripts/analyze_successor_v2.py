"""Recompute and summarize the verified SPARKING v2 development result.

The preregistered admission rule in results.json is primary. Bootstrap intervals
produced here are explicitly post hoc descriptive summaries. The cluster
bootstrap samples unique Matbench row identifiers so materials appearing in
more than one element holdout remain coupled.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ELEMENTS = ("Cu", "Ag", "B")
BOOTSTRAP_SEED = 20261021
BOOTSTRAP_REPLICATES = 20_000
FIGURE_TIMESTAMP = datetime(2026, 10, 21, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _percentile_summary(values: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "lower_95_ev": float(lo),
        "upper_95_ev": float(hi),
        "probability_improvement_gt_0": float(np.mean(values > 0.0)),
        "probability_improvement_gt_0_02_ev": float(np.mean(values > 0.02)),
    }


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    out = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    batch = 500
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = values[indices].mean(axis=1)
    return out


def _cluster_bootstrap(
    rows: dict[str, np.ndarray],
    improvements: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    unique_ids = np.unique(np.concatenate([rows[e] for e in ELEMENTS]))
    index = {int(row): i for i, row in enumerate(unique_ids)}
    sums = np.zeros((len(unique_ids), len(ELEMENTS)), dtype=float)
    present = np.zeros_like(sums)
    for j, element in enumerate(ELEMENTS):
        for row, value in zip(rows[element], improvements[element], strict=True):
            i = index[int(row)]
            sums[i, j] += float(value)
            present[i, j] += 1.0

    out = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    probabilities = np.full(len(unique_ids), 1.0 / len(unique_ids))
    batch = 250
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        counts = rng.multinomial(len(unique_ids), probabilities, size=stop - start)
        numerators = counts @ sums
        denominators = counts @ present
        if np.any(denominators == 0):
            raise RuntimeError("cluster bootstrap produced an empty element holdout")
        out[start:stop] = (numerators / denominators).mean(axis=1)
    return out, len(unique_ids)


def _write_table(path: Path, model_rows: list[dict[str, object]]) -> None:
    labels = {
        "ridge": "Ridge",
        "direct_hgb_squared": "Direct HGB",
        "extra_trees_ensemble": "Extra Trees ensemble",
        "catboost_single_seed": "CatBoost single seed",
        "catboost_ensemble": "CatBoost ensemble",
        "risk_gate_t0.05": "Risk gate, $T=0.05$",
        "risk_gate_t0.40": "Selected risk gate, $T=0.40$",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Cu & Ag & B & Macro \\",
        r"\midrule",
    ]
    for row in model_rows:
        model = str(row["model"])
        if model not in labels:
            continue
        values = row["per_element_mae_ev"]
        lines.append(
            f"{labels.get(model, model)} & "
            f"{values['Cu']:.4f} & {values['Ag']:.4f} & {values['B']:.4f} & "
            f"{row['macro_mae_ev']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(
    pdf_path: Path,
    png_path: Path,
    model_rows: list[dict[str, object]],
    per_element_improvement: dict[str, float],
) -> None:
    keep = {
        "extra_trees_ensemble",
        "direct_hgb_squared",
        "catboost_single_seed",
        "catboost_ensemble",
        "risk_gate_t0.05",
        "risk_gate_t0.40",
    }
    shown = [row for row in model_rows if row["model"] in keep]
    label_map = {
        "extra_trees_ensemble": "Extra Trees",
        "direct_hgb_squared": "Direct HGB",
        "catboost_single_seed": "CatBoost\nsingle",
        "catboost_ensemble": "CatBoost\nensemble",
        "risk_gate_t0.05": "Best gate\n$T=0.05$",
        "risk_gate_t0.40": "Selected gate\n$T=0.40$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)
    y_pos = np.arange(len(shown))
    macro = [float(row["macro_mae_ev"]) for row in shown]
    axes[0].barh(y_pos, macro, color="#386cb0", height=0.68)
    axes[0].set_yticks(y_pos, [label_map[str(row["model"])].replace("\n", " ") for row in shown], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Element-macro MAE (eV)")
    axes[0].set_xlim(0.70, 0.80)
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].set_title("Development performance")

    effect_values = [per_element_improvement[e] for e in ELEMENTS]
    colors = ["#4daf4a" if value > 0 else "#e41a1c" for value in effect_values]
    axes[1].bar(ELEMENTS, effect_values, color=colors, width=0.65)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.02, color="#984ea3", linestyle="--", linewidth=1.0, label="admission threshold")
    axes[1].set_ylabel("MAE improvement vs CatBoost (eV)")
    axes[1].set_title("Selected gate, $T=0.40$")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "SPARKING v2 deterministic analysis",
            "Producer": "Matplotlib",
            "CreationDate": FIGURE_TIMESTAMP,
            "ModDate": FIGURE_TIMESTAMP,
        },
    )
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", default="kaggle-development-v2-r1")
    args = parser.parse_args()
    root = args.root.resolve()
    run = root / "runs" / args.run_id
    results_path = run / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selected = str(results["selection"]["selected"]["model"])
    if selected != "risk_gate_t0.40":
        raise RuntimeError(f"unexpected frozen selected model: {selected}")

    rows_by_model: dict[str, dict[str, float]] = {}
    for row in results["rows"]:
        rows_by_model.setdefault(str(row["model"]), {})[str(row["element"])] = float(row["mae_ev"])
    model_rows = [
        {
            "model": model,
            "per_element_mae_ev": values,
            "macro_mae_ev": float(np.mean([values[e] for e in ELEMENTS])),
        }
        for model, values in rows_by_model.items()
    ]
    model_rows.sort(key=lambda row: float(row["macro_mae_ev"]))

    prediction_rows: dict[str, np.ndarray] = {}
    paired_improvements: dict[str, np.ndarray] = {}
    per_element: dict[str, dict[str, object]] = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    declared = results["development_admission"]["per_element_improvement_ev"]
    for element in ELEMENTS:
        arrays = np.load(run / f"{element}-predictions.npz")
        y = arrays["target_ev"]
        baseline = arrays["prediction_catboost_ensemble"]
        gated = arrays[f"prediction_{selected}"]
        improvement = np.abs(y - baseline) - np.abs(y - gated)
        point = float(improvement.mean())
        if not np.isclose(point, float(declared[element]), atol=1e-12, rtol=0.0):
            raise RuntimeError(f"{element} raw paired effect does not match results.json")
        prediction_rows[element] = arrays["row_index"].astype(np.int64)
        paired_improvements[element] = improvement
        boot = _bootstrap_mean(improvement, rng)
        per_element[element] = {
            "n": int(len(y)),
            "paired_mae_improvement_ev": point,
            "paired_row_bootstrap": _percentile_summary(boot),
        }

    macro_point = float(np.mean([per_element[e]["paired_mae_improvement_ev"] for e in ELEMENTS]))
    if not np.isclose(
        macro_point,
        float(results["development_admission"]["macro_mae_improvement_over_catboost_ev"]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("raw macro effect does not match results.json")
    cluster_boot, unique_rows = _cluster_bootstrap(prediction_rows, paired_improvements, rng)

    diagnostics = {}
    for element in ELEMENTS:
        payload = json.loads((run / f"{element}-diagnostics.json").read_text(encoding="utf-8"))
        diagnostics[element] = {
            "pseudo_calibration_rows": int(payload["pseudo_calibration_rows"]),
            "pseudo_harm_fit_spearman": float(payload["pseudo_harm_fit_spearman"]),
            "outer_risk_harm_spearman_descriptive": float(payload["outer_risk_harm_spearman_descriptive"]),
            "catboost_seed_std_mean_ev": float(payload["catboost_seed_std_mean_ev"]),
        }

    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "analysis_status": "VERIFIED_POST_HOC_DESCRIPTIVE",
        "primary_disposition": results["scientific_disposition"],
        "protected_outcomes_observed": bool(results["protected_confirmatory_outcomes_observed"]),
        "selected_model": selected,
        "bootstrap": {
            "status": "POST_HOC_NOT_PREREGISTERED",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "cluster_unit": "unique Matbench row_index across overlapping element holdouts",
            "unique_cluster_count": unique_rows,
        },
        "selected_gate_effect": {
            "macro_mae_improvement_ev": macro_point,
            "cluster_bootstrap": _percentile_summary(cluster_boot),
            "per_element": per_element,
        },
        "model_macro_mae": model_rows,
        "risk_calibration_diagnostics": diagnostics,
        "source_files": {
            "results_json_sha256": _sha256(results_path),
            **{
                f"{element}_predictions_npz_sha256": _sha256(run / f"{element}-predictions.npz")
                for element in ELEMENTS
            },
        },
    }

    artifacts = root / "artifacts"
    figures = root / "paper" / "figures"
    tables = root / "paper" / "tables"
    artifacts.mkdir(exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    analysis_path = artifacts / "development-v2-posthoc-analysis.json"
    analysis_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    _write_table(tables / "development-v2-results.tex", model_rows)
    _write_figure(
        figures / "development-v2-results.pdf",
        figures / "development-v2-results.png",
        model_rows,
        {e: float(per_element[e]["paired_mae_improvement_ev"]) for e in ELEMENTS},
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
