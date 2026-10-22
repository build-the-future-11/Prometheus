import numpy as np
import pytest

from ml4materials.features import CompositionFeaturizer


def test_features_are_formula_invariant() -> None:
    featurizer = CompositionFeaturizer().fit(["Fe2O3", "ZnO"])
    features = featurizer.transform(["Fe2O3", "Fe4O6", "O3Fe2"])
    assert np.array_equal(features[0], features[1])
    assert np.array_equal(features[0], features[2])


def test_featurizer_must_be_fit() -> None:
    with pytest.raises(RuntimeError):
        CompositionFeaturizer().transform(["GaAs"])


def test_feature_schema_is_fixed_but_novelty_state_is_train_only() -> None:
    featurizer = CompositionFeaturizer().fit(["GaAs", "ZnO"])
    assert featurizer.transform(["LiF"]).shape[1] == len(featurizer.feature_names)
    assert featurizer.unseen_elements("LiF") == ("Li", "F")
    assert featurizer.unseen_elements("ZnO") == ()


def test_missing_formula_fails() -> None:
    featurizer = CompositionFeaturizer().fit(["GaAs"])
    with pytest.raises((TypeError, ValueError)):
        featurizer.transform([None])  # type: ignore[list-item]

