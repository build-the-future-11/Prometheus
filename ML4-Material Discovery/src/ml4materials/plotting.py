"""Static, reproducible figures generated only from retained result arrays."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "ml4-material-discovery-cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
# Matplotlib's macOS system-profiler response is malformed in some sandboxed
# environments.  The package already ships DejaVu fonts, which are sufficient
# for our static figures and avoid an unversioned system-font dependency.
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parity_plot(targets: np.ndarray, proposed: np.ndarray, ridge: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 5.4))
    axis.scatter(targets, ridge, s=9, alpha=0.22, label="Ridge baseline", color="#6B7280")
    axis.scatter(targets, proposed, s=9, alpha=0.28, label="Calibrated ExtraTrees", color="#2563EB")
    low = float(min(targets.min(), proposed.min(), ridge.min()))
    high = float(max(targets.max(), proposed.max(), ridge.max()))
    axis.plot([low, high], [low, high], "k--", linewidth=1, label="Ideal")
    axis.set(xlabel="Experimental band gap (eV)", ylabel="Out-of-fold prediction (eV)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    _save(fig, path)


def calibration_plot(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    nominal = [row["nominal_coverage"] for row in rows]
    observed = [row["observed_coverage"] for row in rows]
    axis.plot([0.45, 1.0], [0.45, 1.0], "k--", linewidth=1, label="Ideal")
    axis.plot(nominal, observed, marker="o", color="#7C3AED", label="Split conformal")
    axis.set(
        xlabel="Nominal interval coverage",
        ylabel="Observed out-of-fold coverage",
        xlim=(0.45, 1.0),
        ylim=(0.45, 1.0),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _save(fig, path)


def abstention_plot(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 4.8))
    retained = [row["retained_fraction"] for row in rows]
    mae = [row["retained_mae_ev"] for row in rows]
    axis.plot(retained, mae, marker="o", color="#DC2626")
    axis.invert_xaxis()
    axis.set(xlabel="Fraction retained (lowest uncertainty)", ylabel="MAE (eV)")
    axis.grid(alpha=0.2)
    _save(fig, path)


def ranking_plot(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.9))
    labels = {
        "random": "Random",
        "greedy_predicted_property": "Greedy predicted property",
        "uncertainty": "Uncertainty",
        "proposed_calibrated_probability": "Proposed calibrated probability",
    }
    colors = {
        "random": "#6B7280",
        "greedy_predicted_property": "#059669",
        "uncertainty": "#D97706",
        "proposed_calibrated_probability": "#2563EB",
    }
    for policy in labels:
        selected = sorted(
            (row for row in rows if row["policy"] == policy), key=lambda row: row["budget"]
        )
        axis.plot(
            [row["budget"] for row in selected],
            [row["precision"] for row in selected],
            marker="o",
            label=labels[policy],
            color=colors[policy],
        )
    axis.set(xlabel="Label budget", ylabel="Precision in 1.1–1.6 eV window")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    _save(fig, path)


def subgroup_plot(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [row for row in rows if row["n"] >= 10]
    fig, axis = plt.subplots(figsize=(7.2, 4.9))
    axis.barh(
        [str(row["subgroup"]).replace("_", " ") for row in selected],
        [row["mae_ev"] for row in selected],
        color="#0F766E",
    )
    axis.set(xlabel="Out-of-fold MAE (eV)")
    axis.grid(axis="x", alpha=0.2)
    _save(fig, path)

