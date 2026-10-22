from pathlib import Path

import numpy as np

from ml4materials.models import (
    ensemble_predictions,
    fit_ensemble,
    load_checkpoint,
    make_ridge,
    save_checkpoint,
)
from ml4materials.uncertainty import NormalizedConformalCalibrator


def _config() -> dict[str, float | int]:
    return {
        "n_estimators_per_seed": 8,
        "max_depth": 6,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "n_jobs": 1,
    }


def test_ridge_scaler_is_fit_only_when_model_is_fit() -> None:
    model = make_ridge([0.1, 1.0])
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    targets = np.asarray([0.0, 1.0, 2.0, 3.0])
    model.fit(features, targets)
    assert model.named_steps["scale"].mean_[0] == 1.5


def test_checkpoint_round_trip_is_prediction_identical(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(40, 4))
    targets = features[:, 0] - 0.5 * features[:, 1]
    models = fit_ensemble(features[:30], targets[:30], [1, 2], _config())
    prediction, raw_sigma, _ = ensemble_predictions(models, features[30:])
    calibrator = NormalizedConformalCalibrator.fit(
        targets[20:30],
        ensemble_predictions(models, features[20:30])[0],
        ensemble_predictions(models, features[20:30])[1],
    )
    path = tmp_path / "model.joblib"
    save_checkpoint(
        path,
        {
            "models": models,
            "calibrator": calibrator.to_dict(),
            "feature_names": ["a", "b", "c", "d"],
            "train_element_counts": {},
        },
    )
    loaded = load_checkpoint(path)
    loaded_prediction, loaded_raw_sigma, _ = ensemble_predictions(
        loaded["models"], features[30:]
    )
    assert np.array_equal(prediction, loaded_prediction)
    assert np.array_equal(raw_sigma, loaded_raw_sigma)


def test_label_permutation_degrades_synthetic_signal() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(240, 5))
    targets = 3.0 * features[:, 0] - 2.0 * features[:, 1]
    train, test = np.arange(180), np.arange(180, 240)
    real = fit_ensemble(features[train], targets[train], [7], _config())[0]
    permuted = fit_ensemble(
        features[train], rng.permutation(targets[train]), [7], _config()
    )[0]
    real_mae = np.mean(np.abs(targets[test] - real.predict(features[test])))
    permuted_mae = np.mean(np.abs(targets[test] - permuted.predict(features[test])))
    assert real_mae < permuted_mae

