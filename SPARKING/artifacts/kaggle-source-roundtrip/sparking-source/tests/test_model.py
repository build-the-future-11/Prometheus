import numpy as np

from sparking.model import SparkingRegressor


CONFIG = {
    "max_iter": 40,
    "learning_rate": 0.08,
    "max_leaf_nodes": 15,
    "l2_regularization": 0.1,
    "min_samples_leaf": 5,
    "random_state": 11,
}


def test_gate_is_one_in_support_and_decays_outside():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 6))
    y = 1.2 * x[:, 0] - 0.8 * x[:, 1] + np.sin(x[:, 2])
    model = SparkingRegressor.fit(x, y, CONFIG)
    prediction, gate, distance = model.predict(np.vstack([x[:4], np.full((1, 6), 100.0)]), 0.95, 2.0)
    assert prediction.shape == (5,)
    assert np.all(gate[:4] == 1.0)
    assert gate[-1] < 0.01
    assert distance[-1] > distance[:4].max()


def test_compute_matched_ungated_equals_unit_gate_formula():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(90, 4))
    y = x[:, 0] ** 2 + x[:, 1]
    model = SparkingRegressor.fit(x, y, CONFIG)
    baseline, residual, _ = model.components(x[:10])
    expected = np.clip(baseline + residual, 0.0, None)
    np.testing.assert_allclose(model.predict_ungated(x[:10]), expected)

