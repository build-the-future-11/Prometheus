import numpy as np

from ml4materials.evaluation import (
    abstention_metrics,
    calibration_metrics,
    regression_metrics,
    subgroup_metrics,
)
from ml4materials.uncertainty import NormalizedConformalCalibrator


def test_regression_metrics_are_exact_for_perfect_prediction() -> None:
    targets = np.asarray([0.0, 1.0, 2.0, 3.0])
    metrics = regression_metrics(targets, targets.copy())
    assert metrics == {"mae_ev": 0.0, "rmse_ev": 0.0, "r2": 1.0, "spearman": 1.0}


def test_abstention_and_calibration_have_prespecified_rows() -> None:
    targets = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    predictions = np.asarray([0.1, 1.1, 2.2, 2.8, 4.5])
    sigma = np.asarray([0.1, 0.1, 0.2, 0.3, 0.8])
    abstention = abstention_metrics(targets, predictions, sigma, [1.0, 0.4])
    assert abstention[0]["rejected_mae_ev"] is None
    assert abstention[1]["retained_count"] == 2
    calibrator = NormalizedConformalCalibrator.fit(targets, predictions, sigma)
    calibration = calibration_metrics(
        targets, predictions, sigma, calibrator, [0.5, 0.9]
    )
    assert [row["nominal_coverage"] for row in calibration] == [0.5, 0.9]
    assert all(row["mean_interval_width_ev"] >= 0 for row in calibration)


def test_subgroups_distinguish_formula_and_novel_element_cases() -> None:
    targets = np.asarray([1.0, 2.0, 3.0, 4.0])
    predictions = np.asarray([1.1, 2.2, 2.8, 4.1])
    rows = subgroup_metrics(
        targets,
        predictions,
        ["FeO", "GaAs", "LiFeO2", "BaTiO3"],
        np.asarray([0, 1, 0, 0]),
    )
    names = {row["subgroup"] for row in rows}
    assert {"binary", "ternary", "contains_oxygen", "no_oxygen", "has_unseen_element"} <= names

