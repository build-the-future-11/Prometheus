import numpy as np
import pytest

from ml4materials.robust import aggregate, conformal_radius, grouped_folds, proper_calibration, paired_bootstrap, quantile_collapse_diagnostic


def test_same_tree_aggregation_has_hand_verifiable_outlier_behavior():
    predictions = np.array([[0., 0., 12.], [1., 2., 3.]])
    np.testing.assert_array_equal(aggregate(predictions, "median"), [0, 2])
    np.testing.assert_array_equal(aggregate(predictions, "mean"), [4, 2])
    with pytest.raises(ValueError):
        aggregate(np.array([[np.nan]]), "median")


def test_conformal_order_statistic_and_unbounded_small_sample():
    assert conformal_radius(np.arange(1., 10.), .9) == 9
    assert conformal_radius(np.arange(1., 10.), .95) == float("inf")
    assert conformal_radius(np.arange(1., 20.), .95) == 19
    with pytest.raises(ValueError):
        conformal_radius(np.array([-1.]), .5)


def test_hash_folds_are_disjoint_balanced_and_label_blind():
    keys = [f"unit-{i}" for i in range(103)]
    folds = grouped_folds(keys, 11)
    assert np.bincount(folds).max() - np.bincount(folds).min() == 1
    np.testing.assert_array_equal(folds, grouped_folds(keys, 11))
    assert not np.array_equal(folds, grouped_folds(keys, 12))
    for fold in range(5):
        outer, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit, cal = proper_calibration(keys, outer, 731, .2)
        assert not (set(fit) & set(cal) or set(fit) & set(test) or set(cal) & set(test))
        assert set(fit) | set(cal) | set(test) == set(range(103))
    with pytest.raises(ValueError):
        grouped_folds(["same", "same"], 11, 2)


def test_bootstrap_does_not_multiply_independent_units_by_repeats():
    differences = np.array([[-1., 0., 1.], [-1., 0., 1.]])
    a = paired_bootstrap(differences, 100, 7)
    b = paired_bootstrap(differences[:1], 100, 7)
    assert a == b
    assert a["mean_difference_ev"] == 0


def test_quantile_diagnostic_retains_both_initializations():
    diagnostic = quantile_collapse_diagnostic()
    assert set(diagnostic["models"]) == {"default_median", "positive_initialization"}
    # This test checks recording, not a version-dependent expected scientific outcome.
    for model in diagnostic["models"].values():
        assert model["first_tree_nodes"] >= 1
        assert np.isfinite(model["training_mae_ev"])
