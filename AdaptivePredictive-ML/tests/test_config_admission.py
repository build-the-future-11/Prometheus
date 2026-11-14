"""Regression cases for invalid configurations previously admitted silently."""

from dataclasses import replace

import numpy as np
import pytest

from adaptive_predictive.model import DDFSConfig, DriftDirectedFixedShare


@pytest.mark.parametrize("name", ["ridge", "eta", "base_share", "max_share", "drift_gain", "fast_rate", "slow_rate", "prediction_clip", "loss_clip", "min_scale", "interval_alpha"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "invalid"])
def test_scalar_hyperparameters_are_finite_reals(name, value):
    with pytest.raises(ValueError):
        DriftDirectedFixedShare(season=4, config=replace(DDFSConfig(), **{name: value}))


@pytest.mark.parametrize("changes", [
    {"lags": ()}, {"lags": (1.0, 2.0)}, {"lags": (True,)},
    {"interval_window": 0}, {"interval_window": -1}, {"interval_window": 2.5}, {"interval_window": True},
    {"min_scale": 0.0}, {"min_scale": -1.0}, {"loss_clip": 0.0}, {"prediction_clip": -1.0},
    {"drift_gain": -1.0}, {"forgetting_factors": (True,)}, {"forgetting_factors": (np.nan,)},
])
def test_invalid_structural_hyperparameters_rejected(changes):
    with pytest.raises(ValueError):
        DriftDirectedFixedShare(season=4, config=replace(DDFSConfig(), **changes))


@pytest.mark.parametrize("season", [0, -1, 1.5, True, np.nan, np.inf])
def test_season_requires_positive_integer(season):
    with pytest.raises(ValueError):
        DriftDirectedFixedShare(season=season)


def test_numpy_integer_configuration_remains_valid():
    model = DriftDirectedFixedShare(season=np.int64(4), config=DDFSConfig(lags=(np.int64(1),), interval_window=np.int64(8)))
    assert model.minimum_history == 8
