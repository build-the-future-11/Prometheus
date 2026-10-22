import pytest
from scripts.verify_evidence import close, metrics, interval, manifest


def test_independent_metrics_match_hand_calculation():
    rows = [{"y": "0", "p": "1", "lo": "0", "hi": "1"},
            {"y": "3", "p": "1", "lo": "0", "hi": "2"}]
    assert metrics(rows, "y", "p")["mae_ev"] == 1.5
    result = interval(rows, "y", "lo", "hi", .5)
    assert result["observed_coverage"] == .5
    assert result["mean_interval_score_ev"] == 3.5
    with pytest.raises(ValueError):
        close(1, 2, "corrupted number")
    with pytest.raises(ValueError):
        metrics([{"y": "0", "p": "nan"}], "y", "p")


def test_manifest_refuses_unmanifested_and_corrupt_content(tmp_path):
    import hashlib
    import json
    (tmp_path / "prediction.csv").write_text("original")
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({"files": {
        "prediction.csv": {"bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}}}))
    assert manifest(tmp_path) == 1
    (tmp_path / "prediction.csv").write_text("tampered")
    with pytest.raises(ValueError, match="artifact mismatch"):
        manifest(tmp_path)
