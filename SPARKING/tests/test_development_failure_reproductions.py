"""Small mechanistic reproducers; these do not evaluate protected materials."""

import numpy as np
from sklearn._loss.loss import AbsoluteError

from sparking.model import make_hgb


CONFIG = {
    "max_iter": 40, "learning_rate": 0.1, "max_leaf_nodes": 7,
    "min_samples_leaf": 5, "l2_regularization": 0.1, "random_state": 11,
}


def zero_majority_fixture():
    x = np.linspace(0, 1, 120).reshape(-1, 1)
    y = np.where(x[:, 0] < .6, 0., 2. + x[:, 0])
    return x, y


def test_v1_absolute_loss_stalls_on_zero_majority_fixture():
    x, y = zero_majority_fixture()
    initial = np.full_like(y, np.median(y))
    assert np.all(AbsoluteError().gradient(y, initial) == -1)
    model = make_hgb(CONFIG).fit(x, y)
    np.testing.assert_array_equal(model.predict(x), initial)
    # Tree topology can differ across sklearn versions through numerical split
    # ties; the regression is failure to move away from the constant median.


def test_squared_loss_avoids_stall_with_the_same_tree_budget():
    x, y = zero_majority_fixture()
    model = make_hgb(CONFIG).set_params(loss="squared_error").fit(x, y)
    prediction = model.predict(x)
    baseline_mae = np.mean(np.abs(y - np.median(y)))
    assert np.mean(np.abs(y - prediction)) < baseline_mae * .1
    assert np.ptp(prediction) > 2
    assert model.n_iter_ == CONFIG["max_iter"]
