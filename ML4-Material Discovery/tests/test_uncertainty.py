import numpy as np

from ml4materials.uncertainty import (
    NormalizedConformalCalibrator,
    calibration_partition,
    normal_target_probability,
)


def test_calibration_partition_is_deterministic_and_group_isolated() -> None:
    identifiers = ["a", "b", "c", "d", "e", "f"]
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g3", "e": "g4", "f": "g5"}
    first = calibration_partition(identifiers, 0.25, 7, groups)
    second = calibration_partition(identifiers, 0.25, 7, groups)
    assert first == second
    train, calibration = first
    assert set(train).isdisjoint(calibration)
    assert ({"a", "b"} <= set(train)) or ({"a", "b"} <= set(calibration))


def test_conformal_intervals_and_probability_are_finite() -> None:
    targets = np.asarray([1.0, 2.0, 1.5, 2.5, 0.5])
    predictions = np.asarray([1.1, 1.8, 1.4, 2.2, 0.7])
    raw_sigma = np.asarray([0.1, 0.2, 0.1, 0.3, 0.2])
    calibrator = NormalizedConformalCalibrator.fit(targets, predictions, raw_sigma)
    lower, upper, quantile = calibrator.interval(predictions, raw_sigma, 0.8)
    assert np.all(lower <= predictions)
    assert np.all(upper >= predictions)
    assert quantile > 0
    probability = normal_target_probability(
        predictions, calibrator.calibrated_sigma(raw_sigma), 1.1, 1.6
    )
    assert np.all((probability >= 0) & (probability <= 1))

