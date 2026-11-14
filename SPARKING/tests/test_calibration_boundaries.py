"""Boundary and malformed-evidence tests for the calibration repair."""

import numpy as np
import pytest

from sparking.metrics import bootstrap_mean_interval, conformal_radius, interval_metrics, regression_metrics


def test_unattainable_finite_sample_rank_is_infinite():
    assert np.isposinf(conformal_radius(np.array([0.2, 0.4]), 0.9))
    assert np.isposinf(conformal_radius(np.arange(8.0), 0.9))


def test_attainable_boundary_uses_last_observation():
    assert conformal_radius(np.arange(9.0), 0.9) == 8.0
    assert conformal_radius(np.arange(19.0), 0.95) == 18.0


@pytest.mark.parametrize("errors", [[], [1.0], [-1.0, 2.0], [1.0, np.nan], [1.0, np.inf], [[1.0, 2.0]]])
def test_invalid_calibration_residuals_rejected(errors):
    with pytest.raises(ValueError):
        conformal_radius(np.asarray(errors), 0.9)


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_invalid_coverage_rejected(coverage):
    with pytest.raises(ValueError):
        conformal_radius(np.arange(10.0), coverage)
    with pytest.raises(ValueError):
        interval_metrics(np.array([1.0]), np.array([0.0]), np.array([2.0]), coverage)


@pytest.mark.parametrize("target,prediction", [([], []), ([1.0], []), ([[1.0]], [[1.0]]), ([np.nan], [1.0])])
def test_regression_rejects_missing_or_invalid_evidence(target, prediction):
    with pytest.raises(ValueError):
        regression_metrics(np.asarray(target), np.asarray(prediction))


@pytest.mark.parametrize("target,lower,upper", [([], [], []), ([1.0], [-np.inf], [np.inf]), ([np.nan], [0.0], [2.0]), ([[1.0]], [[0.0]], [[2.0]]), ([1.0], [2.0], [0.0])])
def test_finite_interval_summary_rejects_invalid_or_unbounded_input(target, lower, upper):
    with pytest.raises(ValueError):
        interval_metrics(np.asarray(target), np.asarray(lower), np.asarray(upper), 0.9)


@pytest.mark.parametrize("resamples", [0, -1, True, 1.5])
def test_invalid_bootstrap_budget_rejected(resamples):
    with pytest.raises(ValueError):
        bootstrap_mean_interval(np.array([1.0, 2.0]), resamples=resamples)
