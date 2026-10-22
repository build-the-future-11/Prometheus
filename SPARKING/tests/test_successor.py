import numpy as np
import pytest

from sparking.successor import convex_prediction, element_frequency_sample_weights, risk_gate_weights


def test_risk_gate_is_unit_weight_for_nonpositive_predicted_harm():
    harm = np.asarray([-1.0, -0.1, 0.0])
    np.testing.assert_array_equal(risk_gate_weights(harm, 0.1), np.ones(3))


def test_risk_gate_monotonically_shrinks_positive_harm():
    weights = risk_gate_weights(np.asarray([0.0, 0.1, 0.2]), 0.1)
    assert weights[0] == 1.0
    assert weights[0] > weights[1] > weights[2] > 0.0


def test_convex_prediction_respects_endpoints_and_bounds():
    ridge = np.asarray([1.0, -1.0, 2.0])
    direct = np.asarray([3.0, 4.0, -2.0])
    np.testing.assert_allclose(convex_prediction(ridge, direct, 1.0), np.asarray([3.0, 4.0, 0.0]))
    np.testing.assert_allclose(convex_prediction(ridge, direct, 0.0), np.asarray([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        convex_prediction(ridge, direct, 1.1)


def test_risk_gate_rejects_invalid_temperature_and_nonfinite_input():
    with pytest.raises(ValueError, match="positive"):
        risk_gate_weights(np.asarray([0.1]), 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        risk_gate_weights(np.asarray([np.nan]), 0.1)


def test_frequency_weights_reduce_exactly_to_unweighted_control():
    formulas = ["Li2O", "NaCl", "LiF"]
    np.testing.assert_array_equal(element_frequency_sample_weights(formulas, 0.0), np.ones(3))


def test_frequency_weights_emphasize_rare_element_compositions():
    formulas = ["Li2O", "LiF", "LiCl", "NaBr"]
    weights = element_frequency_sample_weights(formulas, 1.0, minimum=0.01, maximum=100.0)
    assert weights[-1] > weights[0]
    assert float(np.mean(weights)) == pytest.approx(1.0)
    assert np.all(weights > 0.0)


def test_frequency_weights_reject_invalid_contract():
    with pytest.raises(ValueError, match="at least one"):
        element_frequency_sample_weights([], 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        element_frequency_sample_weights(["LiF"], 1.1)
