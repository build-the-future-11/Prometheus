import numpy as np
import pytest

from sparking.metrics import bootstrap_mean_interval, conformal_radius, interval_metrics, regression_metrics


def test_regression_metrics_hand_verifiable():
    result = regression_metrics(np.array([0.0, 2.0]), np.array([1.0, 0.0]))
    assert result["mae_ev"] == pytest.approx(1.5)
    assert result["rmse_ev"] == pytest.approx(np.sqrt(2.5))
    assert result["bias_ev"] == pytest.approx(-0.5)


def test_finite_sample_conformal_quantile_and_interval_score():
    assert conformal_radius(np.arange(1.0, 11.0), 0.8) == 9.0
    result = interval_metrics(np.array([1.0, 3.0]), np.array([0.0, 0.0]), np.array([2.0, 2.0]), 0.8)
    assert result["observed_coverage"] == 0.5
    assert result["mean_width_ev"] == 2.0


def test_bootstrap_is_deterministic():
    first = bootstrap_mean_interval(np.array([1.0, 2.0, 3.0]), resamples=100)
    second = bootstrap_mean_interval(np.array([1.0, 2.0, 3.0]), resamples=100)
    assert first == second

