import inspect

import numpy as np

from ml4materials.ranking import evaluate_ranking, policy_orders


def test_policy_construction_has_no_label_parameter() -> None:
    assert "targets" not in inspect.signature(policy_orders).parameters


def test_ranking_policies_use_equal_budgets() -> None:
    targets = np.asarray([1.2, 1.4, 0.0, 3.0, 1.5, 2.5])
    predictions = np.asarray([1.2, 1.3, 0.2, 2.8, 1.4, 2.4])
    sigma = np.asarray([0.1, 0.2, 0.4, 0.2, 0.1, 0.3])
    rows, order, probability = evaluate_ranking(
        targets, predictions, sigma, (1.1, 1.6), [2], 16
    )
    assert len(order) == len(targets)
    assert len(probability) == len(targets)
    assert {row["policy"] for row in rows} == {
        "random",
        "greedy_predicted_property",
        "uncertainty",
        "proposed_calibrated_probability",
    }
    assert {row["budget"] for row in rows} == {2}

