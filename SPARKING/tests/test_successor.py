import numpy as np
import pytest

from sparking.successor import convex_prediction, risk_gate_weights


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
