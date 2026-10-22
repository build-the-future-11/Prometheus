"""Deterministic experiment runner and evidence writer."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .model import DDFSConfig, DriftDirectedFixedShare, SeasonalNaive


MODES = (
    "seasonal_naive",
    "uniform",
    "hedge",
    "adahedge",
    "fixed_share",
    "adaptive_share",
    "directed_fixed_share",
    "ddfs",
)


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    values: tuple[float, ...]
    season: int
    source: str
    source_sha256: str
    timestamps: tuple[str, ...] | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def run_series(spec: SeriesSpec, config: DDFSConfig, *, evaluate_from: int) -> dict[str, object]:
    if evaluate_from < max(2 * spec.season, 16) or evaluate_from >= len(spec.values):
        raise ValueError("evaluate_from must leave adequate warm-up and evaluation data")
    models: dict[str, object] = {"seasonal_naive": SeasonalNaive(spec.season)}
    for mode in MODES[1:]:
        models[mode] = DriftDirectedFixedShare(season=spec.season, config=config, mode=mode)  # type: ignore[arg-type]

    rows: list[dict[str, object]] = []
    for index, outcome in enumerate(spec.values):
        if not all(model.ready() for model in models.values()):  # type: ignore[attr-defined]
            for model in models.values():
                model.observe_initial(outcome)  # type: ignore[attr-defined]
            continue
        forecasts: dict[str, float] = {}
        diagnostics: dict[str, object] = {}
        for name, model in models.items():
            prediction = model.predict()  # type: ignore[attr-defined]
            if name == "seasonal_naive":
                forecasts[name] = float(prediction)
            else:
                forecasts[name] = prediction.value
                diagnostics[name] = {
                    "weights": prediction.weights,
                    "drift_score": prediction.drift_score,
                    "share": prediction.share,
                    "lower": prediction.lower,
                    "upper": prediction.upper,
                }
        if index >= evaluate_from:
            row: dict[str, object] = {"index": index, "outcome": outcome, "forecast": forecasts}
            if spec.timestamps is not None:
                row["timestamp"] = spec.timestamps[index]
            row["diagnostics"] = diagnostics
            rows.append(row)
        for model in models.values():
            model.update(outcome)  # type: ignore[attr-defined]

    training_values = np.asarray(spec.values[:evaluate_from], dtype=np.float64)
    mase_scale = float(np.mean(np.abs(training_values[spec.season :] - training_values[: -spec.season])))
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise ValueError("training MASE scale is invalid")
    metrics: dict[str, dict[str, float]] = {}
    outcomes = np.asarray([row["outcome"] for row in rows], dtype=np.float64)
    for name in MODES:
        predictions = np.asarray([row["forecast"][name] for row in rows], dtype=np.float64)  # type: ignore[index]
        errors = predictions - outcomes
        metrics[name] = {
            "mae": float(np.mean(np.abs(errors))),
            "mase": float(np.mean(np.abs(errors)) / mase_scale),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "bias": float(np.mean(errors)),
        }
        if name != "seasonal_naive":
            diagnostics_for_mode = [row["diagnostics"][name] for row in rows]  # type: ignore[index]
            covered = [
                item["lower"] <= outcome <= item["upper"]
                for item, outcome in zip(diagnostics_for_mode, outcomes, strict=True)
            ]
            widths = [item["upper"] - item["lower"] for item in diagnostics_for_mode]
            metrics[name]["interval_coverage"] = float(np.mean(covered))
            metrics[name]["mean_interval_width"] = float(np.mean(widths))
    return {
        "series": spec.name,
        "source": spec.source,
        "source_sha256": spec.source_sha256,
        "season": spec.season,
        "evaluate_from": evaluate_from,
        "n_evaluated": len(rows),
        "mase_scale": mase_scale,
        "first_timestamp": spec.timestamps[0] if spec.timestamps else None,
        "last_timestamp": spec.timestamps[-1] if spec.timestamps else None,
        "metrics": metrics,
        "predictions": rows,
    }


def synthetic_series(length: int = 1200, seed: int = 20261021) -> SeriesSpec:
    """Sanity fixture with declared regime changes; never real-world evidence."""
    rng = np.random.default_rng(seed)
    values = np.empty(length, dtype=np.float64)
    values[:24] = 20 + 3 * np.sin(2 * np.pi * np.arange(24) / 24) + rng.normal(0, 0.4, 24)
    for index in range(24, length):
        regime = 0 if index < length * 0.45 else (1 if index < length * 0.72 else 2)
        level = (20.0, 27.0, 17.0)[regime]
        amplitude = (3.0, 5.0, 2.0)[regime]
        persistence = (0.25, 0.65, -0.10)[regime]
        seasonal = level + amplitude * np.sin(2 * np.pi * index / 24)
        prior_seasonal = level + amplitude * np.sin(2 * np.pi * (index - 1) / 24)
        values[index] = seasonal + persistence * (values[index - 1] - prior_seasonal) + rng.normal(0, 0.5)
    data = canonical_json({"seed": seed, "values": values.tolist()})
    return SeriesSpec(
        name="synthetic-regime-fixture",
        values=tuple(float(item) for item in values),
        season=24,
        source="generated:synthetic_series",
        source_sha256=sha256_bytes(data),
    )


def write_run(output: Path, results: Iterable[dict[str, object]], config: DDFSConfig, *, stage: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    results = list(results)
    manifest = {
        "schema_version": "adaptive-predictive.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "config": asdict(config),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "series": [
            {
                key: result[key]
                for key in ("series", "source", "source_sha256", "season", "evaluate_from", "n_evaluated")
            }
            for result in results
        ],
    }
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    summary = {result["series"]: result["metrics"] for result in results}
    (output / "summary.json").write_bytes(canonical_json(summary))
    with (output / "predictions.jsonl").open("wb") as handle:
        for result in results:
            for row in result["predictions"]:  # type: ignore[union-attr]
                handle.write(canonical_json({"series": result["series"], **row}))
    files = ["manifest.json", "summary.json", "predictions.jsonl"]
    checksums = {name: sha256_bytes((output / name).read_bytes()) for name in files}
    (output / "checksums.json").write_bytes(canonical_json(checksums))


def verify_run(output: Path) -> list[str]:
    issues: list[str] = []
    required = ("manifest.json", "summary.json", "predictions.jsonl", "checksums.json")
    for name in required:
        if not (output / name).is_file():
            issues.append(f"missing:{name}")
    if issues:
        return issues
    checksums = json.loads((output / "checksums.json").read_text())
    for name, expected in checksums.items():
        actual = sha256_bytes((output / name).read_bytes())
        if actual != expected:
            issues.append(f"hash-mismatch:{name}")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema_version") != "adaptive-predictive.run.v1":
        issues.append("unsupported-schema")
    summary = json.loads((output / "summary.json").read_text())
    if not summary:
        issues.append("empty-summary")
    prediction_count = sum(1 for line in (output / "predictions.jsonl").read_text().splitlines() if line)
    expected_count = sum(item["n_evaluated"] for item in manifest.get("series", []))
    if prediction_count != expected_count:
        issues.append("prediction-denominator-mismatch")
    return issues

