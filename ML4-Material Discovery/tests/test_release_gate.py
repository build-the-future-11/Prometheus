from pathlib import Path

from ml4materials.experiment import create_artifact_manifest
from scripts.reproduce_submission import verify_release_candidate
from scripts.reproduce_submission import REQUIRED_OUTPUTS


def test_release_gate_refuses_missing_and_incomplete_runs(tmp_path: Path) -> None:
    assert verify_release_candidate(tmp_path, "missing")
    run = tmp_path / "runs" / "failed"
    run.mkdir(parents=True)
    (run / "receipt.json").write_text('{"status": "FAILED_RETAINED"}', encoding="utf-8")
    errors = verify_release_candidate(tmp_path, "failed")
    assert any("not 'COMPLETED'" in error for error in errors)


def test_release_gate_accepts_completed_manifest_valid_run(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "complete"
    for relative in sorted(REQUIRED_OUTPUTS - {"artifact_manifest.json"}):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "receipt.json":
            path.write_text('{"status": "COMPLETED"}\n', encoding="utf-8")
        elif path.suffix == ".json":
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_bytes(b"fixture\n")
    create_artifact_manifest(run)
    assert verify_release_candidate(tmp_path, "complete") == []
