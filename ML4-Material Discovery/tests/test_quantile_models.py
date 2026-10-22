import numpy as np
import pytest

from ml4materials.evaluation import interval_metrics
from ml4materials.models import (
    fit_quantile_interval_models,
    load_quantile_checkpoint,
    make_quantile_gradient_boosting,
    quantile_interval_predictions,
    save_quantile_checkpoint,
)


def test_quantile_models_produce_ordered_finite_interval() -> None:
    features = np.arange(80, dtype=float).reshape(-1, 1) / 10.0
    targets = 0.4 * features[:, 0] + np.sin(features[:, 0])
    lower, upper = fit_quantile_interval_models(
        features, targets, 0.8, {"n_estimators": 20, "seed": 7}
    )
    low, high = quantile_interval_predictions(lower, upper, features)
    assert np.all(np.isfinite(low))
    assert np.all(low <= high)
    metrics = interval_metrics(targets, low, high, 0.8)
    assert metrics["mean_interval_width_ev"] >= 0.0
    assert metrics["mean_interval_score_ev"] >= metrics["mean_interval_width_ev"]


def test_quantile_inputs_and_interval_metrics_fail_closed() -> None:
    with pytest.raises(ValueError, match="quantile"):
        make_quantile_gradient_boosting(1.0, {})
    with pytest.raises(ValueError, match="coverage"):
        interval_metrics(np.ones(2), np.zeros(2), np.ones(2), 1.0)


def test_quantile_checkpoint_round_trip(tmp_path) -> None:
    features = np.arange(20, dtype=float).reshape(-1, 1)
    targets = features[:, 0] / 10.0
    lower, upper = fit_quantile_interval_models(features, targets, 0.8, {"n_estimators": 10})
    point = make_quantile_gradient_boosting(0.5, {"n_estimators": 10})
    point.fit(features, targets)
    path = tmp_path / "quantile.joblib"
    save_quantile_checkpoint(
        path,
        {
            "point_model": point,
            "interval_models": {"0.8": {"lower": lower, "upper": upper}},
            "feature_names": ["x"],
            "train_element_counts": {"H": 1},
        },
    )
    loaded = load_quantile_checkpoint(path)
    assert np.array_equal(loaded["point_model"].predict(features), point.predict(features))
