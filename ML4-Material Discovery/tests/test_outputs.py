from pathlib import Path

import numpy as np
import pytest

from ml4materials.data import MaterialRecord
from ml4materials.experiment import (
    _candidate_pool,
    _pooled_calibration,
    create_artifact_manifest,
    verify_artifact_manifest,
)
from ml4materials.plotting import (
    abstention_plot,
    calibration_plot,
    parity_plot,
    ranking_plot,
    subgroup_plot,
)
from ml4materials.report import render_manuscript, render_results


def test_candidate_pool_collapses_duplicates_and_combines_uncertainty() -> None:
    records = [
        MaterialRecord("a", "TeSe", 1.58, "Se:1|Te:1"),
        MaterialRecord("b", "Te0.5Se0.5", 1.0, "Se:1|Te:1"),
        MaterialRecord("c", "GaAs", 1.4, "Ga:1|As:1"),
    ]
    metadata, targets, predictions, sigma = _candidate_pool(
        records,
        np.asarray(["fold_0", "fold_1", "fold_2"], dtype=object),
        np.asarray([1.58, 1.0, 1.4]),
        np.asarray([1.5, 1.1, 1.3]),
        np.asarray([0.2, 0.3, 0.1]),
    )
    assert len(metadata) == len(targets) == 2
    duplicate = next(row for row in metadata if row["source_row_count"] == 2)
    assert duplicate["source_target_span_ev"] == pytest.approx(0.58)
    assert np.all(sigma > 0)
    assert np.all(np.isfinite(predictions))


def test_pooled_calibration_and_artifact_manifest(tmp_path: Path) -> None:
    targets = np.asarray([1.0, 2.0])
    rows = _pooled_calibration(
        targets,
        {0.9: (np.asarray([0.5, 1.5]), np.asarray([1.5, 2.5]))},
    )
    assert rows[0]["observed_coverage"] == 1.0
    (tmp_path / "evidence.txt").write_text("stable", encoding="utf-8")
    create_artifact_manifest(tmp_path)
    assert verify_artifact_manifest(tmp_path)["ok"] is True
    (tmp_path / "evidence.txt").write_text("changed", encoding="utf-8")
    result = verify_artifact_manifest(tmp_path)
    assert result["ok"] is False
    assert result["problems"] == ["hash mismatch: evidence.txt"]


def _synthetic_results() -> tuple[dict, dict]:
    metric = {"mae_ev": 0.4, "rmse_ev": 0.6, "r2": 0.5, "spearman": 0.7}
    ranking = []
    for policy, precision in [
        ("random", 0.2),
        ("greedy_predicted_property", 0.5),
        ("uncertainty", 0.1),
        ("proposed_calibrated_probability", 0.6),
    ]:
        ranking.append(
            {
                "policy": policy,
                "budget": 50,
                "hits": precision * 50,
                "precision": precision,
                "enrichment_vs_pool": precision / 0.2,
            }
        )
    results = {
        "official_cv": {
            "metrics": {"proposed": metric, "ridge": {**metric, "mae_ev": 0.5}, "trivial": {**metric, "mae_ev": 1.0}},
            "calibration": [{"nominal_coverage": 0.9, "observed_coverage": 0.89, "mean_interval_width_ev": 1.2}],
            "abstention": [{"retained_mae_ev": 0.4}, {"retained_mae_ev": 0.2}],
        },
        "candidate_ranking": {"pool_size": 100, "metrics": ranking},
        "generalization": [{"held_out_element": "Li", "train_n": 80, "test_n": 20, "proposed": {**metric, "mae_ev": 0.8}, "ridge": metric, "coverage_90": 0.85}],
        "sanity_checks": {"label_permutation": {"mean_mae_ev": 1.2, "unpermuted_fold_0_mae_ev": 0.4, "passes": True}, "checkpoint_reload_all_passed": True},
    }
    receipt = {
        "run_id": "fixture",
        "started_utc": "2026-01-01T00:00:00Z",
        "finished_utc": "2026-01-01T00:00:01Z",
        "duration_seconds": 1.0,
        "python": "3.12",
        "inputs": {"dataset_sha256": "a", "split_sha256": "b"},
        "resources": {"peak_observed_rss_mib": 100.0, "max_allowed_rss_mib": 512.0, "pageout_delta": 0},
    }
    return results, receipt


def test_reports_and_figures_render_from_results(tmp_path: Path) -> None:
    results, receipt = _synthetic_results()
    assert "EXECUTED_DEVELOPMENT" in render_results(results, receipt)
    assert "Abstract" in render_manuscript(results, receipt)

    targets = np.asarray([0.0, 1.0, 2.0])
    parity_plot(targets, targets + 0.1, targets + 0.2, tmp_path / "parity.png")
    calibration_plot(
        [{"nominal_coverage": 0.9, "observed_coverage": 0.85}],
        tmp_path / "calibration.png",
    )
    abstention_plot(
        [{"retained_fraction": 1.0, "retained_mae_ev": 0.5}, {"retained_fraction": 0.5, "retained_mae_ev": 0.3}],
        tmp_path / "abstention.png",
    )
    ranking_plot(results["candidate_ranking"]["metrics"], tmp_path / "ranking.png")
    subgroup_plot(
        [{"subgroup": "binary", "n": 20, "mae_ev": 0.4}],
        tmp_path / "subgroups.png",
    )
    assert len(list(tmp_path.glob("*.png"))) == 5

