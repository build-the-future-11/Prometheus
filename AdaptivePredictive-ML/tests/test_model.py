from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_predictive.model import DDFSConfig, DriftDirectedFixedShare


def warmed_model(mode: str = "ddfs") -> DriftDirectedFixedShare:
    model = DriftDirectedFixedShare(season=4, mode=mode)  # type: ignore[arg-type]
    for value in range(model.minimum_history):
        model.observe_initial(10.0 + math.sin(value))
    return model


def test_predict_then_update_contract() -> None:
    model = warmed_model()
    model.predict()
    with pytest.raises(RuntimeError, match="previous prediction"):
        model.predict()
    model.update(11.0)
    model.predict()


def test_current_outcome_cannot_enter_prediction() -> None:
    left = warmed_model()
    right = warmed_model()
    assert left.predict() == right.predict()
    left.update(-1000.0)
    right.update(1000.0)
    assert left.predict().value != right.predict().value


def test_modes_share_identical_initial_expert_values() -> None:
    predictions = [
        warmed_model(mode).predict()
        for mode in (
            "uniform",
            "hedge",
            "adahedge",
            "fixed_share",
            "adaptive_share",
            "directed_fixed_share",
            "ddfs",
        )
    ]
    assert all(prediction.expert_values == predictions[0].expert_values for prediction in predictions)


def test_modes_keep_identical_expert_predictions_over_time() -> None:
    modes = ("uniform", "hedge", "adahedge", "fixed_share", "adaptive_share", "directed_fixed_share", "ddfs")
    models = [warmed_model(mode) for mode in modes]
    for index in range(40):
        predictions = [model.predict() for model in models]
        assert all(prediction.expert_values == predictions[0].expert_values for prediction in predictions)
        outcome = 10.0 + math.sin(index / 3.0)
        for model in models:
            model.update(outcome)


def test_mechanism_controls_share_identical_detector_state() -> None:
    modes = ("fixed_share", "adaptive_share", "directed_fixed_share", "ddfs")
    models = [warmed_model(mode) for mode in modes]
    for index in range(40):
        predictions = [model.predict() for model in models]
        outcome = 10.0 + 3.0 * math.sin(index / 4.0)
        for model in models:
            model.update(outcome)
        assert all(model.fast_loss == models[0].fast_loss for model in models)
        assert all(model.slow_loss == models[0].slow_loss for model in models)
        assert all(np.array_equal(model.expert_fast_loss, models[0].expert_fast_loss) for model in models)
        assert all(np.array_equal(model.expert_slow_loss, models[0].expert_slow_loss) for model in models)


def test_ddfs_directs_share_to_short_memory_under_drift() -> None:
    model = warmed_model()
    model.fast_loss = 2.0
    model.slow_loss = 0.2
    prediction = model.predict()
    model.update(prediction.value + 20 * prediction.scale)
    assert model.weights[0] > model.weights[-1]
    assert np.isclose(model.weights.sum(), 1.0)


def test_ddfs_target_prefers_recently_better_adapting_expert() -> None:
    model = warmed_model()
    target = model._share_target(1.0, expert_fast=np.array([0.2, 0.8, 1.2]), expert_slow=np.ones(3))
    assert target[0] > target[1] > target[2]


def test_mechanism_controls_isolate_share_rate_and_target() -> None:
    models = {
        mode: warmed_model(mode)
        for mode in ("fixed_share", "adaptive_share", "directed_fixed_share", "ddfs")
    }
    for model in models.values():
        model.fast_loss = 2.0
        model.slow_loss = 0.2
        model.expert_fast_loss = np.array([0.2, 0.8, 1.2])
        model.expert_slow_loss = np.ones(3)

    fixed_share = models["fixed_share"]
    adaptive_share = models["adaptive_share"]
    directed_fixed = models["directed_fixed_share"]
    ddfs = models["ddfs"]
    score = 1.0

    assert fixed_share._share(score) == directed_fixed._share(score)
    assert adaptive_share._share(score) == ddfs._share(score)
    assert adaptive_share._share(score) > fixed_share._share(score)

    uniform_target = fixed_share._share_target(score)
    assert np.allclose(adaptive_share._share_target(score), uniform_target)
    assert np.allclose(directed_fixed._share_target(score), ddfs._share_target(score))
    assert not np.allclose(ddfs._share_target(score), uniform_target)


def test_adahedge_is_finite_normalized_and_parameter_free() -> None:
    model = warmed_model("adahedge")
    assert np.allclose(model.weights, np.full(3, 1.0 / 3.0))
    for offset in (0.0, 2.0, -1.0, 4.0):
        prediction = model.predict()
        model.update(prediction.expert_values[0] + offset * prediction.scale)
        assert np.all(np.isfinite(model.weights))
        assert np.all(model.weights >= 0.0)
        assert np.isclose(model.weights.sum(), 1.0)
        assert model.mixability_gap >= 0.0


def test_numerical_safeguards_keep_predictions_finite() -> None:
    model = warmed_model()
    for index in range(500):
        prediction = model.predict()
        assert math.isfinite(prediction.value)
        outcome = 1e6 if index % 37 == 0 else 10 + math.sin(index)
        model.update(outcome)


@pytest.mark.parametrize(
    "config",
    [
        DDFSConfig(forgetting_factors=()),
        DDFSConfig(forgetting_factors=(0.0,)),
        DDFSConfig(lags=(2, 1)),
        DDFSConfig(fast_rate=0.01, slow_rate=0.1),
    ],
)
def test_invalid_configs_rejected(config: DDFSConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()

