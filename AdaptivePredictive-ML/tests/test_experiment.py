from __future__ import annotations

import json

from adaptive_predictive.experiment import run_series, synthetic_series, verify_run, write_run
from adaptive_predictive.model import DDFSConfig


def test_synthetic_run_is_deterministic() -> None:
    config = DDFSConfig()
    left = run_series(synthetic_series(400, 7), config, evaluate_from=200)
    right = run_series(synthetic_series(400, 7), config, evaluate_from=200)
    assert left["metrics"] == right["metrics"]
    assert left["predictions"] == right["predictions"]


def test_run_bundle_round_trip(tmp_path) -> None:
    config = DDFSConfig()
    result = run_series(synthetic_series(400, 9), config, evaluate_from=200)
    output = tmp_path / "run"
    write_run(output, [result], config, stage="test")
    assert verify_run(output) == []
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["series"][0]["n_evaluated"] == 200


def test_verifier_detects_tampering(tmp_path) -> None:
    config = DDFSConfig()
    result = run_series(synthetic_series(400, 11), config, evaluate_from=200)
    output = tmp_path / "run"
    write_run(output, [result], config, stage="test")
    with (output / "summary.json").open("a") as handle:
        handle.write(" ")
    assert "hash-mismatch:summary.json" in verify_run(output)

