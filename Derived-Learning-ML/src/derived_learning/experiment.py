"""Experiment orchestration, selection boundaries, accounting, and manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .linalg import normalized_mse
from .model import GroupedLinearModel
from .rules import (
    AdamRule,
    AnalyticResidualRule,
    BlockJacobiResidualRule,
    BlockGradientAblationRule,
    ClassicalBackfittingRule,
    FixedNormalizedLMSRule,
    GlobalRidgeRule,
    ResourceCounter,
    training_loss,
)
from .tasks import LearningTask, family_from_dict, generate_from_config, write_tasks


@dataclass
class Checkpoint:
    step: int
    arithmetic_work_units: int
    support_examples_read: int
    train_mse: float
    evaluation_nmse: float
    weights: list[float] | None = None
    intercept: float | None = None


@dataclass
class TrainingResult:
    task_id: str
    split: str
    family: str
    seed: int
    method: str
    hyperparameters: dict[str, float]
    status: str
    error: str | None
    checkpoints: list[dict[str, Any]]
    resources: dict[str, int]
    final_weights: list[float]
    final_intercept: float = 0.0


def _checkpoint_record(checkpoint: Checkpoint) -> dict[str, Any]:
    """Keep S1's checkpoint wire format unless a successor asks for traces."""

    record = asdict(checkpoint)
    for field in ("weights", "intercept"):
        if record[field] is None:
            del record[field]
    return record


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Reject malformed or scientifically ambiguous experiment protocols."""

    if config.get("schema_version") != "derived-learning.experiment-config.v1":
        raise ValueError("unsupported experiment config schema")
    if not isinstance(config.get("study_id"), str) or not config["study_id"].strip():
        raise ValueError("config must declare a non-empty study_id")
    required_splits = {"meta_train", "development", "final"}
    if not required_splits.issubset(config.get("splits", {})):
        raise ValueError("config must declare meta_train, development, and final splits")
    budgets = config.get("budgets", {})
    required_budgets = {"selection_work_units", "final_work_units"}
    if not required_budgets.issubset(budgets) or any(
        not isinstance(budgets[name], int) or isinstance(budgets[name], bool) or budgets[name] <= 0
        for name in required_budgets
    ):
        raise ValueError("work-unit budgets must be positive")
    if "long_horizon" in config["splits"] and (
        not isinstance(budgets.get("long_horizon_work_units"), int)
        or isinstance(budgets.get("long_horizon_work_units"), bool)
        or budgets["long_horizon_work_units"] <= 0
    ):
        raise ValueError("long_horizon_work_units must be positive when that split is declared")
    candidates = config.get("candidate_grid", {})
    fixed = config.get("fixed_conditions", {})
    if not candidates or not candidates.keys().isdisjoint(fixed):
        raise ValueError("candidate_grid must be non-empty and disjoint from fixed_conditions")
    declared = set(candidates) | set(fixed)
    if "analytic_residual" not in declared or "adam" not in declared:
        raise ValueError("config must include analytic_residual and adam for the primary comparison")
    known_methods = {
        "adam", "analytic_residual", "block_jacobi_residual",
        "block_gradient_no_derivation", "fixed_normalized_lms",
        "classical_backfitting", "global_ridge",
    }
    unknown = declared - known_methods
    if unknown:
        raise ValueError(f"unknown configured methods: {sorted(unknown)}")
    if not isinstance(config.get("selection", {}).get("meta_top_k"), int) or config["selection"]["meta_top_k"] <= 0:
        raise ValueError("selection.meta_top_k must be a positive integer")
    criteria = config.get("success_criteria", {})
    for field in ("quality_threshold_nmse", "minimum_relative_equal_resource_improvement"):
        value = criteria.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"success_criteria.{field} must be finite")
    seen_seeds: dict[int, str] = {}
    for split, payload in config["splits"].items():
        if split not in {"meta_train", "development", "final", "long_horizon"}:
            raise ValueError(f"unsupported split: {split}")
        seeds = payload.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            raise ValueError(f"split {split} must declare a non-empty integer seed list")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"split {split} contains duplicate seeds")
        for seed in seeds:
            if seed in seen_seeds:
                raise ValueError(f"seed {seed} is reused across {seen_seeds[seed]} and {split}")
            seen_seeds[seed] = split
        family_from_dict(payload.get("family", {}))


def _make_rule(method: str, hyperparameters: dict[str, float], dimension: int, *, fit_intercept: bool = False):
    if method == "adam":
        return AdamRule(dimension, learning_rate=hyperparameters["learning_rate"], fit_intercept=fit_intercept)
    if method == "analytic_residual":
        return AnalyticResidualRule(regularization=hyperparameters["regularization"])
    if method == "block_jacobi_residual":
        return BlockJacobiResidualRule(regularization=hyperparameters["regularization"])
    if method == "fixed_normalized_lms":
        return FixedNormalizedLMSRule(step_size=hyperparameters["step_size"])
    if method == "block_gradient_no_derivation":
        return BlockGradientAblationRule(learning_rate=hyperparameters["learning_rate"])
    if method == "classical_backfitting":
        return ClassicalBackfittingRule(regularization=hyperparameters["regularization"])
    if method == "global_ridge":
        return GlobalRidgeRule(regularization=hyperparameters["regularization"], fit_intercept=fit_intercept)
    raise ValueError(f"unknown method: {method}")


def run_training(
    task: LearningTask,
    method: str,
    hyperparameters: dict[str, float],
    work_budget: int,
    *,
    record_checkpoint_weights: bool = False,
    fit_intercept: bool = False,
) -> TrainingResult:
    dimension = len(task.support.features[0])
    model = GroupedLinearModel.zeros(dimension, task.groups, fit_intercept=fit_intercept)
    rule = _make_rule(method, hyperparameters, dimension, fit_intercept=fit_intercept)
    resources = ResourceCounter()
    checkpoints: list[dict[str, Any]] = []
    status = "SUCCEEDED"
    error: str | None = None
    try:
        initial = Checkpoint(
            step=0,
            arithmetic_work_units=0,
            support_examples_read=0,
            train_mse=training_loss(model, task.support.features, task.support.targets),
            evaluation_nmse=normalized_mse(model.predict(task.evaluation.features), task.evaluation.targets),
            weights=list(model.weights) if record_checkpoint_weights else None,
            intercept=model.intercept if record_checkpoint_weights and fit_intercept else None,
        )
        checkpoints.append(_checkpoint_record(initial))
        step = 0
        while resources.arithmetic_work_units < work_budget:
            step += 1
            if step > 10_000:
                raise RuntimeError("safety stop: update count exceeded 10,000")
            rule.step(model, task.support.features, task.support.targets, resources)
            train_value = training_loss(model, task.support.features, task.support.targets)
            evaluation_value = normalized_mse(model.predict(task.evaluation.features), task.evaluation.targets)
            if not math.isfinite(train_value) or not math.isfinite(evaluation_value):
                raise FloatingPointError("non-finite metric")
            checkpoints.append(
                _checkpoint_record(
                    Checkpoint(
                        step=step,
                        arithmetic_work_units=resources.arithmetic_work_units,
                        support_examples_read=resources.support_examples_read,
                        train_mse=train_value,
                        evaluation_nmse=evaluation_value,
                        weights=list(model.weights) if record_checkpoint_weights else None,
                        intercept=model.intercept if record_checkpoint_weights and fit_intercept else None,
                    )
                )
            )
            if bool(getattr(rule, "converged", False)):
                break
    except Exception as exception:  # retain every failed attempt as result data
        status = "FAILED"
        error = f"{type(exception).__name__}: {exception}"
    return TrainingResult(
        task_id=task.task_id,
        split=task.split,
        family=task.family,
        seed=task.seed,
        method=method,
        hyperparameters=dict(hyperparameters),
        status=status,
        error=error,
        checkpoints=checkpoints,
        resources=resources.to_dict(),
        final_weights=list(model.weights),
        final_intercept=model.intercept,
    )


def _candidate_key(parameters: dict[str, float]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _terminal_score(result: TrainingResult) -> float:
    if result.status != "SUCCEEDED" or not result.checkpoints:
        return math.inf
    return float(result.checkpoints[-1]["evaluation_nmse"])


def select_hyperparameters(
    config: dict[str, Any],
    tasks: list[LearningTask],
    *,
    record_checkpoint_weights: bool = False,
    fit_intercept: bool = False,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]], dict[str, Any]]:
    selection_budget = int(config["budgets"]["selection_work_units"])
    top_k = int(config["selection"]["meta_top_k"])
    records: list[dict[str, Any]] = []
    selected: dict[str, dict[str, float]] = {}
    rationale: dict[str, Any] = {}
    by_split = {
        split: [task for task in tasks if task.split == split]
        for split in ("meta_train", "development")
    }
    for method, candidates in config["candidate_grid"].items():
        meta_means: dict[str, float] = {}
        for candidate in candidates:
            key = _candidate_key(candidate)
            scores: list[float] = []
            for task in by_split["meta_train"]:
                result = run_training(task, method, candidate, selection_budget, record_checkpoint_weights=record_checkpoint_weights, fit_intercept=fit_intercept)
                record = asdict(result)
                records.append(record)
                scores.append(_terminal_score(result))
            meta_means[key] = statistics.fmean(scores)
        retained_keys = sorted(meta_means, key=lambda key: (meta_means[key], key))[: min(top_k, len(candidates))]
        retained = [candidate for candidate in candidates if _candidate_key(candidate) in retained_keys]
        development_means: dict[str, float] = {}
        for candidate in retained:
            key = _candidate_key(candidate)
            scores = []
            for task in by_split["development"]:
                result = run_training(task, method, candidate, selection_budget, record_checkpoint_weights=record_checkpoint_weights, fit_intercept=fit_intercept)
                records.append(asdict(result))
                scores.append(_terminal_score(result))
            development_means[key] = statistics.fmean(scores)
        winning_key = min(development_means, key=lambda key: (development_means[key], key))
        selected[method] = json.loads(winning_key)
        rationale[method] = {
            "meta_train_mean_nmse": meta_means,
            "retained_after_meta_train": retained_keys,
            "development_mean_nmse": development_means,
            "selected": selected[method],
            "selection_rule": "lowest mean terminal development NMSE after predeclared meta-train top-k filter",
        }
    return selected, records, rationale


def _last_at_or_below(result: dict[str, Any], budget: int) -> dict[str, Any]:
    eligible = [checkpoint for checkpoint in result["checkpoints"] if checkpoint["arithmetic_work_units"] <= budget]
    return eligible[-1] if eligible else result["checkpoints"][0]


def _first_at_threshold(result: dict[str, Any], threshold: float) -> int | None:
    for checkpoint in result["checkpoints"]:
        if checkpoint["evaluation_nmse"] <= threshold:
            return int(checkpoint["arithmetic_work_units"])
    return None


def summarize_results(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    threshold = float(config["success_criteria"]["quality_threshold_nmse"])
    successful = [result for result in results if result["status"] == "SUCCEEDED"]
    methods = sorted({result["method"] for result in results})
    task_ids = sorted({result["task_id"] for result in results})
    terminal_by_pair = {(result["task_id"], result["method"]): result for result in successful}
    paired_tasks = [task_id for task_id in task_ids if all((task_id, method) in terminal_by_pair for method in methods)]
    common_budget_by_task = {
        task_id: min(
            int(terminal_by_pair[(task_id, method)]["checkpoints"][-1]["arithmetic_work_units"])
            for method in methods
        )
        for task_id in paired_tasks
    }
    aggregate: dict[str, Any] = {}
    for method in methods:
        method_results = [result for result in successful if result["method"] == method]
        terminal_scores = [float(result["checkpoints"][-1]["evaluation_nmse"]) for result in method_results]
        equal_scores = [
            float(_last_at_or_below(terminal_by_pair[(task_id, method)], common_budget_by_task[task_id])["evaluation_nmse"])
            for task_id in paired_tasks
        ]
        threshold_costs = [
            cost
            for result in method_results
            if (cost := _first_at_threshold(result, threshold)) is not None
        ]
        aggregate[method] = {
            "tasks_succeeded": len(method_results),
            "tasks_failed": len([result for result in results if result["method"] == method and result["status"] != "SUCCEEDED"]),
            "terminal_mean_nmse": statistics.fmean(terminal_scores) if terminal_scores else None,
            "terminal_population_std_nmse": statistics.pstdev(terminal_scores) if len(terminal_scores) > 1 else 0.0 if terminal_scores else None,
            "equal_resource_mean_nmse": statistics.fmean(equal_scores) if equal_scores else None,
            "threshold_reached_count": len(threshold_costs),
            "median_work_units_to_threshold": statistics.median(threshold_costs) if threshold_costs else None,
            "mean_terminal_work_units": statistics.fmean(
                int(result["checkpoints"][-1]["arithmetic_work_units"]) for result in method_results
            ) if method_results else None,
        }
    proposal = "analytic_residual"
    baseline = "adam"
    paired_deltas: list[float] = []
    proposal_wins = 0
    for task_id in paired_tasks:
        budget = common_budget_by_task[task_id]
        proposal_score = float(_last_at_or_below(terminal_by_pair[(task_id, proposal)], budget)["evaluation_nmse"])
        baseline_score = float(_last_at_or_below(terminal_by_pair[(task_id, baseline)], budget)["evaluation_nmse"])
        paired_deltas.append(proposal_score - baseline_score)
        proposal_wins += int(proposal_score < baseline_score)
    baseline_equal = aggregate.get(baseline, {}).get("equal_resource_mean_nmse")
    proposal_equal = aggregate.get(proposal, {}).get("equal_resource_mean_nmse")
    relative_improvement = None
    if baseline_equal not in (None, 0.0) and proposal_equal is not None:
        relative_improvement = (baseline_equal - proposal_equal) / baseline_equal
    baseline_cost = aggregate.get(baseline, {}).get("median_work_units_to_threshold")
    proposal_cost = aggregate.get(proposal, {}).get("median_work_units_to_threshold")
    quality_pass = relative_improvement is not None and relative_improvement >= float(config["success_criteria"]["minimum_relative_equal_resource_improvement"])
    efficiency_pass = proposal_cost is not None and baseline_cost is not None and proposal_cost < baseline_cost
    return {
        "schema_version": "derived-learning.summary.v1",
        "quality_threshold_nmse": threshold,
        "paired_task_count": len(paired_tasks),
        "common_budget_by_task": common_budget_by_task,
        "methods": aggregate,
        "paired_primary_minus_adam_equal_resource_nmse": paired_deltas,
        "paired_primary_win_count": proposal_wins,
        "relative_equal_resource_improvement_vs_adam": relative_improvement,
        "success_criteria": {
            "quality_pass": quality_pass,
            "efficiency_pass": efficiency_pass,
            "overall_pass": quality_pass and efficiency_pass,
        },
        "interpretation": "SUPPORTED_ON_ADMITTED_MATRIX" if quality_pass and efficiency_pass else "NOT_SUPPORTED_ON_ADMITTED_MATRIX",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_clean(root: Path) -> bool | None:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return status == ""
    except (OSError, subprocess.CalledProcessError):
        return None


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def execute_experiment(config_path: Path, output_dir: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = generate_from_config(config)
    write_tasks(tasks, output_dir / "tasks.json")
    record_checkpoint_weights = bool(config.get("record_checkpoint_weights", False))
    fit_intercept = bool(config.get("fit_intercept", False))
    selected, search_records, rationale = select_hyperparameters(
        config,
        tasks,
        record_checkpoint_weights=record_checkpoint_weights,
        fit_intercept=fit_intercept,
    )
    selection_payload = {
        "schema_version": "derived-learning.selection.v1",
        "uses_final_split": False,
        "selected": selected,
        "rationale": rationale,
    }
    (output_dir / "selection.json").write_text(json.dumps(selection_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(search_records, output_dir / "selection_attempts.jsonl")

    conditions = {**selected, **config["fixed_conditions"]}
    final_records: list[dict[str, Any]] = []
    for task in tasks:
        if task.split not in {"final", "long_horizon"}:
            continue
        budget_name = "long_horizon_work_units" if task.split == "long_horizon" else "final_work_units"
        work_budget = int(config["budgets"][budget_name])
        for method, hyperparameters in conditions.items():
            final_records.append(
                asdict(
                    run_training(
                        task,
                        method,
                        hyperparameters,
                        work_budget,
                        record_checkpoint_weights=record_checkpoint_weights,
                        fit_intercept=fit_intercept,
                    )
                )
            )
    write_jsonl(final_records, output_dir / "results.jsonl")
    summary_by_split = {
        split: summarize_results(config, [record for record in final_records if record["split"] == split])
        for split in sorted({record["split"] for record in final_records})
    }
    summary = {
        "schema_version": "derived-learning.multi-split-summary.v1",
        "study_id": config["study_id"],
        "primary_split": "final",
        "by_split": summary_by_split,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    protocol_path = project_root / config.get("protocol_path", "PROTOCOL.md")
    if not protocol_path.is_file():
        raise ValueError(f"declared protocol is missing: {protocol_path}")
    source_paths = (
        list((project_root / "src").rglob("*.py"))
        + list((project_root / "build_backend").rglob("*.py"))
        + [config_path, protocol_path, project_root / "pyproject.toml"]
    )
    output_files = [
        output_dir / "tasks.json",
        output_dir / "selection.json",
        output_dir / "selection_attempts.jsonl",
        output_dir / "results.jsonl",
        output_dir / "summary.json",
    ]
    manifest = {
        "schema_version": "derived-learning.run-manifest.v1",
        "study_id": config["study_id"],
        "status": "COMPLETE" if all(record["status"] == "SUCCEEDED" for record in final_records) else "COMPLETE_WITH_RETAINED_FAILURES",
        "config_path": str(config_path.relative_to(project_root)),
        "config_sha256": _sha256(config_path),
        "protocol_path": str(protocol_path.relative_to(project_root)),
        "protocol_sha256": _sha256(protocol_path),
        "source_tree_sha256": _tree_hash(source_paths, project_root),
        "git_commit_at_execution": _git_commit(project_root),
        "git_worktree_clean_at_execution": _git_worktree_clean(project_root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not-set; experiment does not use hash iteration for RNG"),
            "third_party_runtime_dependencies": [],
        },
        "split_separation": {
            "meta_train_task_ids": [task.task_id for task in tasks if task.split == "meta_train"],
            "development_task_ids": [task.task_id for task in tasks if task.split == "development"],
            "final_task_ids": [task.task_id for task in tasks if task.split == "final"],
            "long_horizon_task_ids": [task.task_id for task in tasks if task.split == "long_horizon"],
            "final_used_for_selection": False,
        },
        "resource_scope": {
            "paid_services": False,
            "remote_compute": False,
            "worker_count": 1,
            "teacher_calls": 0,
            "learned_target_generator_training_steps": 0,
        },
        "outputs": {str(path.relative_to(output_dir)): _sha256(path) for path in output_files},
        "selected_hyperparameters": selected,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest

