import csv
from pathlib import Path

from ml4materials.features import CompositionFeaturizer
from ml4materials.inference import inspect_checkpoint, predict_formulas, rank_csv
from ml4materials.models import ensemble_predictions, fit_ensemble, save_checkpoint
from ml4materials.uncertainty import NormalizedConformalCalibrator
import numpy as np


def _checkpoint(path: Path) -> None:
    formulas = ["GaAs", "ZnO", "Si", "Fe2O3", "LiF", "BaTiO3", "CdS", "AlN"]
    targets = np.asarray([1.4, 3.3, 1.1, 2.0, 12.0, 3.0, 2.4, 6.0])
    featurizer = CompositionFeaturizer().fit(formulas[:6])
    features = featurizer.transform(formulas)
    config = {
        "n_estimators_per_seed": 8,
        "max_depth": 6,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "n_jobs": 1,
    }
    models = fit_ensemble(features[:6], targets[:6], [1, 2], config)
    calibration_prediction, calibration_sigma, _ = ensemble_predictions(models, features[6:])
    calibrator = NormalizedConformalCalibrator.fit(
        targets[6:], calibration_prediction, calibration_sigma
    )
    save_checkpoint(
        path,
        {
            "models": models,
            "calibrator": calibrator.to_dict(),
            "feature_names": list(featurizer.feature_names),
            "train_element_counts": featurizer.train_element_counts,
        },
    )


def test_checkpoint_prediction_and_label_blind_csv_ranking(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.joblib"
    _checkpoint(checkpoint)
    predictions = predict_formulas(checkpoint, ["GaAs", "XeCl2"])
    assert len(predictions) == 2
    assert predictions[0]["composition"] == "GaAs"
    assert predictions[1]["unseen_elements"] == ["Cl", "Xe"]
    assert 0 <= predictions[0]["target_window_probability"] <= 1

    input_path = tmp_path / "candidates.csv"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "composition"])
        writer.writeheader()
        writer.writerows(
            [
                {"candidate_id": "a", "composition": "GaAs"},
                {"candidate_id": "b", "composition": "ZnO"},
            ]
        )
    output_path = tmp_path / "ranked.csv"
    rank_csv(checkpoint, input_path, output_path)
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert [int(row["rank"]) for row in rows] == [1, 2]
    assert {row["candidate_id"] for row in rows} == {"a", "b"}


def test_checkpoint_inspection_is_schema_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.joblib"
    _checkpoint(checkpoint)
    report = inspect_checkpoint(checkpoint, limit=2)
    assert report["checkpoint_type"] == "extra_trees_conformal"
    assert len(report["top_features"]) == 2

