import json
from pathlib import Path

import pytest

from ml4materials.data import write_json
from ml4materials.studies import (
    build_run_registry,
    compare_completed_runs,
    config_sha256,
    write_preflight_receipt,
)


def _run(root: Path, name: str, *, complete: bool) -> None:
    run = root / "runs" / name
    run.mkdir(parents=True)
    write_json(run / "payload.json", {"name": name})
    from ml4materials.experiment import create_artifact_manifest

    create_artifact_manifest(run)
    receipt = {
        "run_id": name,
        "status": "COMPLETED" if complete else "FAILED_RETAINED",
        "inputs": {"dataset_sha256": "dataset", "split_sha256": "splits"},
    }
    write_json(run / "receipt.json", receipt)
    if complete:
        write_json(
            run / "results.json",
            {
                "status": "EXECUTED_DEVELOPMENT_NOT_INDEPENDENTLY_VALIDATED",
                "study_version": "fixture",
                "official_cv": {"metrics": {"proposed": {"mae_ev": 0.4}}},
            },
        )
    create_artifact_manifest(run)


def test_config_sha256_requires_valid_json(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    config.write_text(json.dumps({"study_version": "fixture"}), encoding="utf-8")
    assert len(config_sha256(config)) == 64
    config.write_text("not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        config_sha256(config)


def test_registry_and_comparison_refuse_incomplete_runs(tmp_path: Path) -> None:
    _run(tmp_path, "complete-a", complete=True)
    _run(tmp_path, "complete-b", complete=True)
    _run(tmp_path, "failed", complete=False)
    registry = build_run_registry(tmp_path)
    assert [entry["run_id"] for entry in registry["runs"]] == [
        "complete-a",
        "complete-b",
        "failed",
    ]
    comparison = compare_completed_runs(tmp_path, ["complete-a", "complete-b"])
    assert len(comparison["runs"]) == 2
    with pytest.raises(ValueError, match="incomplete"):
        compare_completed_runs(tmp_path, ["complete-a", "failed"])


def test_preflight_receipt_retains_refusal(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    config.write_text(json.dumps({"study_version": "fixture"}), encoding="utf-8")
    output = tmp_path / "preflight.json"
    payload = write_preflight_receipt(tmp_path, config, output)
    assert payload["status"] == "REFUSED_AT_PREFLIGHT"
    assert json.loads(output.read_text(encoding="utf-8"))["exception_type"] == "KeyError"
