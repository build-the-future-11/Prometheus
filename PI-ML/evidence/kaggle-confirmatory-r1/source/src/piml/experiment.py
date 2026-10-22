"""Versioned, resumable experiment runner with evidence-bound artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns, time
from typing import Any

from .metrics import case_metrics
from .model import EnsembleOperator, RidgeOperator, jacobi_correct, pcg_solve
from .pde import Case, generate_split, solve_case
from .statistics import bootstrap_mean_ci, mean, median, quantile, sample_sd


CONDITIONS = (
    "direct_thomas",
    "ridge_only",
    "ridge_jacobi",
    "zero_pcg",
    "mean_pcg",
    "ridge_pcg",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_config(config: dict[str, Any]) -> None:
    required = {"study_version", "phase", "grid_points", "modes", "data", "model", "solver", "statistics", "decision"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    if config["phase"] not in {"development", "confirmatory"}:
        raise ValueError("phase must be development or confirmatory")
    data = config["data"]
    seed_ranges: list[set[int]] = []
    for name in ("train", "calibration", "iid_test", "ood_test"):
        spec = data[name]
        if int(spec["count"]) < 1:
            raise ValueError(f"{name} count must be positive")
        seeds = set(range(int(spec["seed"]), int(spec["seed"]) + int(spec["count"])))
        if any(seeds & prior for prior in seed_ranges):
            raise ValueError("dataset seed ranges overlap")
        seed_ranges.append(seeds)
    if config["phase"] == "confirmatory" and not config.get("frozen", False):
        raise ValueError("confirmatory configuration must be explicitly frozen")


def _cases(config: dict[str, Any], name: str) -> list[Case]:
    common = config["data"]["common"]
    spec = config["data"][name]
    return generate_split(
        name,
        int(spec["seed"]),
        int(spec["count"]),
        int(config["grid_points"]),
        float(spec.get("coefficient_scale", common["coefficient_scale"])),
        float(spec.get("source_scale", common["source_scale"])),
        int(config["modes"]),
    )


def _model_json(model: EnsembleOperator) -> dict[str, object]:
    return {
        "members": [
            {"weights": member.weights, "ridge": member.ridge, "grid_points": member.grid_points}
            for member in model.members
        ]
    }


def _calibration_factor(model: EnsembleOperator, cases: list[Case]) -> float:
    scores: list[float] = []
    for case in cases:
        prediction, stddev = model.predict(case)
        scores.extend(
            abs(value - truth) / max(scale, 1e-12)
            for value, truth, scale in zip(prediction[1:-1], case.solution[1:-1], stddev[1:-1])
        )
    # Split-conformal finite-sample rank, capped at the largest observed score.
    probability = min(1.0, (int(0.9 * (len(scores) + 1) + 0.999999) / len(scores)))
    return quantile(scores, probability)


def _average_solution(cases: list[Case]) -> tuple[float, ...]:
    return tuple(sum(values) / len(values) for values in zip(*(case.solution for case in cases)))


def _row(
    case: Case,
    condition: str,
    prediction: tuple[float, ...],
    elapsed_ns: int,
    stddev: tuple[float, ...] | None,
    calibration: float | None,
    iterations: int = 0,
    operator_applications: int = 0,
    converged: bool = True,
    initial_relative_residual: float | None = None,
    final_relative_residual: float | None = None,
    breakdown: str | None = None,
) -> dict[str, object]:
    values: dict[str, object] = {
        "case_id": case.case_id,
        "split": case.split,
        "generator_seed": case.generator_seed,
        "coefficient_scale": case.coefficient_scale,
        "condition": condition,
        "elapsed_ns": elapsed_ns,
        "iterations": iterations,
        "operator_applications": operator_applications,
        "converged": converged,
        "initial_relative_residual": initial_relative_residual if initial_relative_residual is not None else "",
        "final_relative_residual": final_relative_residual if final_relative_residual is not None else "",
        "breakdown": breakdown or "",
    }
    values.update(case_metrics(case, prediction, stddev, calibration))
    return values


def _evaluate_case(
    case: Case,
    model: EnsembleOperator,
    mean_solution: tuple[float, ...],
    calibration: float,
    config: dict[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    solver = config["solver"]
    started = perf_counter_ns()
    direct = solve_case(case.diffusivity, case.forcing)
    elapsed = perf_counter_ns() - started
    rows.append(_row(case, "direct_thomas", direct, elapsed, None, None))

    started = perf_counter_ns()
    prediction, stddev = model.predict(case)
    inference_elapsed = perf_counter_ns() - started
    rows.append(_row(case, "ridge_only", prediction, inference_elapsed, stddev, calibration))

    started = perf_counter_ns()
    jacobi = jacobi_correct(
        case, prediction, int(config["controls"]["jacobi_steps"]), float(config["controls"]["jacobi_relaxation"])
    )
    rows.append(_row(case, "ridge_jacobi", jacobi, inference_elapsed + perf_counter_ns() - started, stddev, calibration))

    solve_arguments = {
        "rtol": float(solver["rtol"]),
        "atol": float(solver["atol"]),
        "max_iterations": int(solver["max_iterations"]),
    }
    for condition, initial, overhead in (
        ("zero_pcg", None, 0),
        ("mean_pcg", mean_solution, 0),
        ("ridge_pcg", prediction, inference_elapsed),
    ):
        result = pcg_solve(case, initial, **solve_arguments)
        rows.append(
            _row(
                case,
                condition,
                result.solution,
                overhead + result.elapsed_ns,
                stddev if condition == "ridge_pcg" else None,
                calibration if condition == "ridge_pcg" else None,
                result.iterations,
                result.operator_applications,
                result.converged,
                result.initial_relative_residual,
                result.final_relative_residual,
                result.breakdown,
            )
        )
    return rows


def _summarize(rows: list[dict[str, object]], config: dict[str, Any]) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    metrics = (
        "relative_l2",
        "relative_residual_l2",
        "energy_error",
        "elapsed_ns",
        "iterations",
        "operator_applications",
    )
    for split in ("iid_test", "ood_test"):
        for condition in CONDITIONS:
            selected = [row for row in rows if row["split"] == split and row["condition"] == condition]
            entry: dict[str, object] = {
                "split": split,
                "condition": condition,
                "n": len(selected),
                "converged_fraction": sum(bool(row["converged"]) for row in selected) / len(selected),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in selected]
                entry[f"{metric}_mean"] = mean(values)
                entry[f"{metric}_median"] = median(values)
                entry[f"{metric}_sd"] = sample_sd(values)
            summaries.append(entry)

    paired: dict[str, object] = {}
    for split in ("iid_test", "ood_test"):
        zero = {str(row["case_id"]): row for row in rows if row["split"] == split and row["condition"] == "zero_pcg"}
        proposed = {str(row["case_id"]): row for row in rows if row["split"] == split and row["condition"] == "ridge_pcg"}
        if zero.keys() != proposed.keys():
            raise RuntimeError("paired case identities differ")
        reductions = [
            (float(zero[key]["iterations"]) - float(proposed[key]["iterations"]))
            / max(float(zero[key]["iterations"]), 1.0)
            for key in sorted(zero)
        ]
        effect, lower, upper = bootstrap_mean_ci(
            reductions,
            int(config["statistics"]["bootstrap_seed"]) + (0 if split == "iid_test" else 1),
            int(config["statistics"]["bootstrap_repeats"]),
        )
        paired[split] = {
            "metric": "relative_iteration_reduction_vs_zero_pcg",
            "n": len(reductions),
            "mean": effect,
            "median": median(reductions),
            "ci95_lower": lower,
            "ci95_upper": upper,
            "all_proposed_converged": all(bool(row["converged"]) for row in proposed.values()),
        }
    threshold = config["decision"]
    primary = paired["iid_test"]
    passed = (
        bool(primary["all_proposed_converged"])
        and float(primary["mean"]) >= float(threshold["minimum_mean_reduction"])
        and float(primary["ci95_lower"]) >= float(threshold["minimum_ci_lower"])
    )
    if config["phase"] == "development":
        disposition = "DEVELOPMENT_ONLY"
    elif passed:
        disposition = "SUPPORTED"
    else:
        disposition = "NEGATIVE"
    return {
        "summaries": summaries,
        "paired_primary": paired,
        "decision": {
            "phase": config["phase"],
            "passed": passed,
            "disposition": disposition,
            "rule": threshold,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(sorted(set().union(*(row.keys() for row in rows)) - set(fieldnames)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_cases(path: Path, cases: list[Case]) -> None:
    with path.open("w") as handle:
        for case in cases:
            handle.write(canonical_json(asdict(case)) + "\n")


def _manifest(output_dir: Path, config_digest: str, started_at: float) -> dict[str, object]:
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(output_dir))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return {
        "schema_version": 1,
        "config_sha256": config_digest,
        "started_unix": started_at,
        "completed_unix": time(),
        "files": files,
    }


def run(config: dict[str, Any], output_dir: Path) -> dict[str, object]:
    """Execute a complete immutable run; an existing directory is rejected."""
    validate_config(config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time()
    config_digest = sha256_bytes(canonical_json(config).encode())
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "cpu_count": os.cpu_count(),
            "pid": os.getpid(),
        },
    )
    train = _cases(config, "train")
    calibration_cases = _cases(config, "calibration")
    iid = _cases(config, "iid_test")
    ood = _cases(config, "ood_test")
    all_cases = train + calibration_cases + iid + ood
    if len({case.case_id for case in all_cases}) != len(all_cases):
        raise RuntimeError("case identifiers are not unique")
    if len({case.generator_seed for case in all_cases}) != len(all_cases):
        raise RuntimeError("generator seeds overlap")
    _write_cases(output_dir / "cases.jsonl", all_cases)

    model_spec = config["model"]
    model = EnsembleOperator.fit(
        train, float(model_spec["ridge"]), int(model_spec["ensemble_members"]), int(model_spec["bootstrap_seed"])
    )
    calibration = _calibration_factor(model, calibration_cases)
    write_json(output_dir / "model.json", _model_json(model))
    write_json(output_dir / "calibration.json", {"nominal_coverage": 0.9, "factor": calibration})
    mean_solution = _average_solution(train)

    rows: list[dict[str, object]] = []
    for case in iid + ood:
        rows.extend(_evaluate_case(case, model, mean_solution, calibration, config))
    _write_csv(output_dir / "metrics.csv", rows)
    analysis = _summarize(rows, config)
    write_json(output_dir / "analysis.json", analysis)
    write_json(
        output_dir / "run_receipt.json",
        {
            "status": "COMPLETE",
            "phase": config["phase"],
            "study_version": config["study_version"],
            "config_sha256": config_digest,
            "case_count": len(all_cases),
            "metric_row_count": len(rows),
            "scientific_disposition": analysis["decision"]["disposition"],
        },
    )
    write_json(output_dir / "manifest.json", _manifest(output_dir, config_digest, started_at))
    verification = verify(output_dir)
    write_json(output_dir / "verification.json", verification)
    # Verification is written after the immutable manifest and deliberately not included in it.
    return {"analysis": analysis, "verification": verification, "output_dir": str(output_dir)}


def verify(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.json is missing")
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    for relative, expected in manifest["files"].items():
        path = output_dir / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256_file(path) != expected["sha256"]:
            failures.append(f"digest:{relative}")
        elif path.stat().st_size != expected["bytes"]:
            failures.append(f"size:{relative}")
    config = json.loads((output_dir / "config.json").read_text())
    digest = sha256_bytes(canonical_json(config).encode())
    if digest != manifest["config_sha256"]:
        failures.append("config-digest")
    receipt = json.loads((output_dir / "run_receipt.json").read_text())
    with (output_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = (
        int(config["data"]["iid_test"]["count"])
        + int(config["data"]["ood_test"]["count"])
    ) * len(CONDITIONS)
    if len(rows) != expected_rows or len(rows) != int(receipt["metric_row_count"]):
        failures.append("metric-row-count")
    keys = [(row["case_id"], row["condition"]) for row in rows]
    if len(set(keys)) != len(keys):
        failures.append("duplicate-case-condition")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "verified_file_count": len(manifest["files"]),
        "verified_metric_rows": len(rows),
        "config_sha256": digest,
    }

