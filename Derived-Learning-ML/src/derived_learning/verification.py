"""Independent readback checks for saved evaluation metrics and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .linalg import normalized_mse
from .model import GroupedLinearModel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    checked: dict[str, str] = {}
    for relative, expected in manifest["outputs"].items():
        path = run_dir / relative
        if not path.is_file():
            raise ValueError(f"manifest output is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"manifest hash mismatch for {relative}: {actual} != {expected}")
        checked[relative] = actual
    return {"schema_version": "derived-learning.manifest-verification.v1", "status": "VERIFIED", "checked": checked}


def verify_saved_evaluation(run_dir: Path) -> dict[str, Any]:
    tasks_payload = json.loads((run_dir / "tasks.json").read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in tasks_payload["tasks"]}
    checked: list[dict[str, Any]] = []
    with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            result = json.loads(line)
            if result["status"] != "SUCCEEDED":
                continue
            task = tasks[result["task_id"]]
            features = [[float(value) for value in row] for row in task["evaluation"]["features"]]
            targets = [float(value) for value in task["evaluation"]["targets"]]
            model = GroupedLinearModel(
                dimension=len(result["final_weights"]),
                groups=[[int(index) for index in group] for group in task["groups"]],
                weights=[float(value) for value in result["final_weights"]],
                intercept=float(result.get("final_intercept", 0.0)),
                fit_intercept="final_intercept" in result,
            )
            model.validate()
            recomputed = normalized_mse(model.predict(features), targets)
            recorded = float(result["checkpoints"][-1]["evaluation_nmse"])
            if abs(recomputed - recorded) > 1e-12 * max(1.0, abs(recorded)):
                raise ValueError(
                    f"evaluation mismatch for {result['task_id']}/{result['method']}: {recomputed} != {recorded}"
                )
            checked.append(
                {
                    "task_id": result["task_id"],
                    "method": result["method"],
                    "recorded_nmse": recorded,
                    "recomputed_nmse": recomputed,
                }
            )
            for checkpoint in result["checkpoints"]:
                if "weights" not in checkpoint:
                    continue
                checkpoint_model = GroupedLinearModel(
                    dimension=len(checkpoint["weights"]),
                    groups=[[int(index) for index in group] for group in task["groups"]],
                    weights=[float(value) for value in checkpoint["weights"]],
                    intercept=float(checkpoint.get("intercept", 0.0)),
                    fit_intercept="intercept" in checkpoint,
                )
                checkpoint_model.validate()
                checkpoint_nmse = normalized_mse(checkpoint_model.predict(features), targets)
                recorded_checkpoint_nmse = float(checkpoint["evaluation_nmse"])
                if abs(checkpoint_nmse - recorded_checkpoint_nmse) > 1e-12 * max(1.0, abs(recorded_checkpoint_nmse)):
                    raise ValueError(
                        f"checkpoint evaluation mismatch for {result['task_id']}/{result['method']}/"
                        f"step-{checkpoint['step']}: {checkpoint_nmse} != {recorded_checkpoint_nmse}"
                    )
    return {
        "schema_version": "derived-learning.evaluation-verification.v1",
        "status": "VERIFIED",
        "checked_result_count": len(checked),
        "results": checked,
    }


def write_verification(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

