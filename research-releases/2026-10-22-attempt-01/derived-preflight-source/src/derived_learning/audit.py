"""Post-run baseline, optimality, and claim-admission audits.

These audits never alter a frozen run.  They consume its retained support data,
selected regularization, terminal models, and manifest-bound outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .linalg import normalized_mse, ridge_solution, ridge_solution_with_intercept
from .model import GroupedLinearModel
from .verification import verify_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _kkt_residual(
    rows: list[list[float]],
    targets: list[float],
    weights: list[float],
    intercept: float,
    regularization: float,
    *,
    fit_intercept: bool,
) -> float:
    predictions = [math.fsum(value * weight for value, weight in zip(row, weights, strict=True)) + intercept for row in rows]
    residuals = [prediction - target for prediction, target in zip(predictions, targets, strict=True)]
    gradient = [
        math.fsum(row[index] * residual for row, residual in zip(rows, residuals, strict=True))
        + regularization * weights[index]
        for index in range(len(weights))
    ]
    if fit_intercept:
        gradient.append(math.fsum(residuals))
    return max(abs(value) for value in gradient)


def audit_global_ridge(run_dir: Path, project_root: Path) -> dict[str, Any]:
    """Compare the saved analytic method with its full-feature ridge optimum."""

    run_dir = run_dir.resolve()
    project_root = project_root.resolve()
    verification = verify_manifest(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config_path = project_root / manifest["config_path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    regularization = float(selection["selected"]["analytic_residual"]["regularization"])
    fit_intercept = bool(config.get("fit_intercept", False))
    tasks_payload = json.loads((run_dir / "tasks.json").read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in tasks_payload["tasks"]}
    analytic_results = [
        result for result in _read_jsonl(run_dir / "results.jsonl")
        if result["method"] == "analytic_residual" and result["status"] == "SUCCEEDED"
    ]
    records: list[dict[str, Any]] = []
    for result in analytic_results:
        task = tasks[result["task_id"]]
        support_rows = [[float(value) for value in row] for row in task["support"]["features"]]
        support_targets = [float(value) for value in task["support"]["targets"]]
        evaluation_rows = [[float(value) for value in row] for row in task["evaluation"]["features"]]
        evaluation_targets = [float(value) for value in task["evaluation"]["targets"]]
        if fit_intercept:
            optimum_weights, optimum_intercept = ridge_solution_with_intercept(
                support_rows, support_targets, regularization
            )
        else:
            optimum_weights = ridge_solution(support_rows, support_targets, regularization)
            optimum_intercept = 0.0
        optimum_model = GroupedLinearModel(
            dimension=len(optimum_weights),
            groups=[[int(index) for index in group] for group in task["groups"]],
            weights=optimum_weights,
            intercept=optimum_intercept,
            fit_intercept=fit_intercept,
        )
        terminal_weights = [float(value) for value in result["final_weights"]]
        terminal_intercept = float(result.get("final_intercept", 0.0))
        distance = math.sqrt(
            math.fsum((left - right) ** 2 for left, right in zip(terminal_weights, optimum_weights, strict=True))
            + (terminal_intercept - optimum_intercept) ** 2
        )
        optimum_norm = math.sqrt(math.fsum(value * value for value in optimum_weights) + optimum_intercept**2)
        records.append(
            {
                "task_id": result["task_id"],
                "split": result["split"],
                "regularization": regularization,
                "analytic_terminal_nmse": float(result["checkpoints"][-1]["evaluation_nmse"]),
                "global_ridge_nmse": normalized_mse(optimum_model.predict(evaluation_rows), evaluation_targets),
                "parameter_l2_distance_to_global_ridge": distance,
                "relative_parameter_distance_to_global_ridge": distance / max(optimum_norm, 1e-15),
                "analytic_kkt_residual_max_abs": _kkt_residual(
                    support_rows, support_targets, terminal_weights, terminal_intercept,
                    regularization, fit_intercept=fit_intercept,
                ),
                "global_ridge_kkt_residual_max_abs": _kkt_residual(
                    support_rows, support_targets, optimum_weights, optimum_intercept,
                    regularization, fit_intercept=fit_intercept,
                ),
            }
        )
    by_split: dict[str, Any] = {}
    for split in sorted({record["split"] for record in records}):
        split_records = [record for record in records if record["split"] == split]
        by_split[split] = {
            "task_count": len(split_records),
            "analytic_terminal_mean_nmse": statistics.fmean(record["analytic_terminal_nmse"] for record in split_records),
            "global_ridge_mean_nmse": statistics.fmean(record["global_ridge_nmse"] for record in split_records),
            "max_relative_parameter_distance": max(record["relative_parameter_distance_to_global_ridge"] for record in split_records),
            "max_analytic_kkt_residual": max(record["analytic_kkt_residual_max_abs"] for record in split_records),
        }
    max_distance = max((record["relative_parameter_distance_to_global_ridge"] for record in records), default=math.inf)
    return {
        "schema_version": "derived-learning.global-ridge-audit.v1",
        "study_id": manifest["study_id"],
        "status": "VERIFIED",
        "manifest_verification_status": verification["status"],
        "inputs": {
            "config_sha256": _sha256(config_path),
            "tasks_sha256": _sha256(run_dir / "tasks.json"),
            "results_sha256": _sha256(run_dir / "results.jsonl"),
            "selection_sha256": _sha256(run_dir / "selection.json"),
        },
        "algorithmic_relation": (
            "analytic_residual is cyclic ridge backfitting/block coordinate minimization of the same "
            "strictly convex full-feature ridge objective; with positive regularization it converges "
            "to the unique global_ridge solution"
        ),
        "numerical_equivalence_status": (
            "EQUIVALENT_WITHIN_1E-8_RELATIVE_PARAMETER_DISTANCE"
            if max_distance <= 1e-8 else "FINITE_BUDGET_NOT_NUMERICALLY_EQUIVALENT"
        ),
        "by_split": by_split,
        "records": records,
    }


def write_global_ridge_audit(run_dir: Path, project_root: Path, output: Path) -> dict[str, Any]:
    payload = audit_global_ridge(run_dir, project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_claim_status(run_dir: Path, baseline_audit: Path | None = None) -> dict[str, Any]:
    """Admit only claims that survive the frozen endpoints and novelty audit."""

    run_dir = run_dir.resolve()
    verification = verify_manifest(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    primary = summary["by_split"][summary["primary_split"]]
    robustness = {
        split: values for split, values in summary["by_split"].items()
        if split != summary["primary_split"]
    }
    primary_pass = bool(primary["success_criteria"]["overall_pass"])
    robustness_pass = bool(robustness) and all(
        values["success_criteria"]["overall_pass"] for values in robustness.values()
    )
    if primary_pass and robustness_pass:
        disposition = "SUPPORTED_WITHIN_FROZEN_SYNTHETIC_MATRIX"
    elif primary_pass:
        disposition = "MIXED"
    else:
        disposition = "NEGATIVE"
    audit_payload = None
    if baseline_audit is not None:
        audit_payload = json.loads(baseline_audit.read_text(encoding="utf-8"))
    blockers = [
        "internally executed synthetic evidence only",
        "no independent reproduction receipt",
        "analytic residual update is classical cyclic ridge backfitting/block coordinate minimization",
    ]
    if not robustness_pass:
        blockers.append("at least one registered robustness or long-horizon endpoint failed")
    if audit_payload is None:
        blockers.append("global ridge optimality audit not supplied")
    return {
        "schema_version": "derived-learning.claim-status.v1",
        "study_id": summary["study_id"],
        "manifest_verification_status": verification["status"],
        "disposition": disposition,
        "broad_superiority_claim_allowed": False,
        "novel_derived_target_mechanism_claim_allowed": False,
        "submission_ready": False,
        "admitted_claim": (
            "The saved study reports its frozen endpoint disposition within its declared synthetic matrix; "
            "the analytic update is an instrumented classical backfitting baseline, not a novel learned target mechanism."
        ),
        "blocked_claims": [
            "general learning superiority",
            "novelty of partial-residual target construction",
            "real-world utility",
            "independent reproduction",
        ],
        "blockers": blockers,
        "baseline_audit": {
            "path": str(baseline_audit) if baseline_audit is not None else None,
            "sha256": _sha256(baseline_audit) if baseline_audit is not None else None,
            "numerical_equivalence_status": audit_payload.get("numerical_equivalence_status") if audit_payload else None,
        },
    }


def write_claim_status(run_dir: Path, output: Path, baseline_audit: Path | None = None) -> dict[str, Any]:
    payload = build_claim_status(run_dir, baseline_audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
