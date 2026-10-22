"""Study planning, preflight checks, and read-only run comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .data import load_data_contract, load_json, sha256_file, write_json
from .experiment import verify_artifact_manifest
from .guards import ResourceGuard


def config_sha256(path: Path) -> str:
    """Return the content hash of a validated JSON experiment configuration."""

    load_json(path)
    return sha256_file(path)


def plan_study(root: Path, config_path: Path) -> dict[str, Any]:
    """Expose a compact, hash-bound plan without allocating a run directory."""

    root = root.resolve()
    config = load_json(config_path)
    contract = load_data_contract(root, str(config["raw_data_contract"]))
    return {
        "schema_version": 1,
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": config_sha256(config_path),
        "study_version": config["study_version"],
        "objective": config["objective"],
        "dataset_sha256": contract["source_sha256"],
        "official_split_sha256": contract["split_source_sha256"],
        "max_rss_mib": config["max_rss_mib"],
        "require_no_pageout_increase": config["require_no_pageout_increase"],
    }


def preflight_study(root: Path, config_path: Path) -> dict[str, Any]:
    """Verify inputs and current resource admission before allocating a run ID."""

    root = root.resolve()
    plan = plan_study(root, config_path)
    config = load_json(config_path)
    contract = load_data_contract(root, str(config["raw_data_contract"]))
    required = [
        root / contract["source_path"],
        root / contract["split_source_path"],
        root / contract["upstream_metadata_path"],
        root / "data/manifests/provenance.json",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"preflight missing required artifacts: {missing}")
    usage = shutil.disk_usage(root)
    guard = ResourceGuard(
        max_allowed_rss_mib=float(config["max_rss_mib"]),
        require_no_pageout_increase=bool(config["require_no_pageout_increase"]),
    )
    observed = guard.check("preflight")
    return {
        **plan,
        "status": "ADMITTED_AT_PREFLIGHT",
        "resources": guard.receipt(),
        "observed": observed,
        "free_disk_bytes": usage.free,
    }


def write_preflight_receipt(
    root: Path, config_path: Path, output_path: Path
) -> dict[str, Any]:
    """Persist both admission and refusal evidence without creating a run."""

    try:
        payload = preflight_study(root, config_path)
    except Exception as exc:
        try:
            config_hash = sha256_file(config_path)
        except OSError:
            config_hash = None
        payload = {
            "schema_version": 1,
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "status": "REFUSED_AT_PREFLIGHT",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    write_json(output_path, payload)
    return payload


def _run_summary(root: Path, run_dir: Path) -> dict[str, Any]:
    receipt_path = run_dir / "receipt.json"
    if not receipt_path.is_file():
        return {"run_id": run_dir.name, "status": "INCOMPLETE_NO_RECEIPT"}
    receipt = load_json(receipt_path)
    summary: dict[str, Any] = {
        "run_id": receipt.get("run_id", run_dir.name),
        "status": receipt.get("status", "UNKNOWN"),
        "started_utc": receipt.get("started_utc"),
        "finished_utc": receipt.get("finished_utc"),
        "dataset_sha256": receipt.get("inputs", {}).get("dataset_sha256"),
        "split_sha256": receipt.get("inputs", {}).get("split_sha256"),
        "artifact_manifest": verify_artifact_manifest(run_dir),
    }
    results_path = run_dir / "results.json"
    if results_path.is_file():
        results = load_json(results_path)
        summary["results_status"] = results.get("status")
        summary["study_version"] = results.get("study_version")
        summary["proposed_mae_ev"] = results.get("official_cv", {}).get("metrics", {}).get(
            "proposed", {}
        ).get("mae_ev")
    return summary


def build_run_registry(root: Path) -> dict[str, Any]:
    """Build a manifest-verified index of retained runs without modifying them."""

    root = root.resolve()
    runs_root = root / "runs"
    entries = [
        _run_summary(root, path)
        for path in sorted(runs_root.iterdir())
        if path.is_dir() and path.name != "registry"
    ]
    return {"schema_version": 1, "runs": entries}


def write_run_registry(root: Path, output_path: Path) -> dict[str, Any]:
    registry = build_run_registry(root)
    write_json(output_path, registry)
    return registry


def compare_completed_runs(root: Path, run_ids: list[str]) -> dict[str, Any]:
    """Compare only manifest-valid, completed runs sharing exact input hashes."""

    if len(run_ids) < 2:
        raise ValueError("compare requires at least two run IDs")
    summaries = [_run_summary(root, root / "runs" / run_id) for run_id in run_ids]
    invalid = [
        summary["run_id"]
        for summary in summaries
        if summary.get("status") != "COMPLETED"
        or not summary.get("artifact_manifest", {}).get("ok")
        or summary.get("proposed_mae_ev") is None
    ]
    if invalid:
        raise ValueError(f"comparison refuses incomplete or unverified runs: {invalid}")
    input_pairs = {
        (summary.get("dataset_sha256"), summary.get("split_sha256")) for summary in summaries
    }
    if len(input_pairs) != 1:
        raise ValueError("comparison refuses runs with different input contracts")
    return {
        "schema_version": 1,
        "input_contract": next(iter(input_pairs)),
        "runs": summaries,
    }
