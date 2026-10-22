import hashlib
import json
from pathlib import Path
import pytest
from scripts.stage_kaggle_replay import FROZEN_BUNDLE_SHA256, stage


def fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "releases/20261021").mkdir(parents=True)
    (tmp_path / "kaggle/reproduction").mkdir(parents=True)
    bundle = tmp_path / "releases/20261021/robust-source.bundle"
    bundle.write_bytes(b"fixture")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    checksum = tmp_path / "releases/20261021/robust-source.sha256"
    checksum.write_text(digest + "  robust-source.bundle\n")
    runner = tmp_path / "kaggle/reproduction/run_reproduction.py"
    runner.write_text('RUN_ID = "kaggle-robust-aggregation-v3-reproduction-r1"\nWORK = Path("/kaggle/working/ml4-material-discovery-v3-reproduction")\n')
    return tmp_path


def test_staging_rewrites_only_the_run_identity(monkeypatch, tmp_path):
    root = fixture_root(tmp_path)
    bundle = root / "releases/20261021/robust-source.bundle"
    actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
    monkeypatch.setattr("scripts.stage_kaggle_replay.FROZEN_BUNDLE_SHA256", actual)
    output = tmp_path / "out"
    manifest = stage(root, "kaggle-robust-aggregation-v3-replay-r2", "ryangomez010/ml4-material-discovery-v3-replay", output)
    assert manifest["status"] == "STAGED_FOR_EXACT_FROZEN_REPLAY"
    assert "replay-r2" in (output / "run_replay.py").read_text()
    assert json.loads((output / "kernel-metadata.json").read_text())["is_private"] == "true"


@pytest.mark.parametrize("run_id", ["replay-r1", "kaggle-robust-aggregation-v3-replay-r0", "kaggle-robust-aggregation-v3-replay-r1/x"])
def test_staging_rejects_invalid_run_ids(run_id, tmp_path):
    with pytest.raises(ValueError):
        stage(tmp_path, run_id, "ryangomez010/ml4-material-discovery-v3-replay", tmp_path / "out")


def test_staging_refuses_existing_output(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError):
        stage(tmp_path, "kaggle-robust-aggregation-v3-replay-r1", "ryangomez010/ml4-material-discovery-v3-replay", output)
