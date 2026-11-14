import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

spec = importlib.util.spec_from_file_location("robustness", Path(__file__).parents[1] / "scripts" / "extend_development_robustness.py")
robustness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robustness)


@pytest.mark.parametrize("values,budget", [([0.0, 1.0], 0.1), ([1.0, 2.0, 3.0], 0.4), ([-0.3, 0.1, 0.1, 0.9], 0.2), ([0.1, 0.1, 0.1], 0.9)])
def test_exact_reweighting_matches_independent_linear_program(values, budget):
    values = np.asarray(values)
    n = len(values)
    # Independent LP with auxiliary absolute deviations: sum |q-u| <= 2*TV.
    objective = np.r_[values, np.zeros(n)]
    constraints = np.vstack([np.c_[np.eye(n), -np.eye(n)], np.c_[-np.eye(n), -np.eye(n)], np.r_[np.zeros(n), np.ones(n)]])
    rhs = np.r_[np.full(n, 1/n), np.full(n, -1/n), 2*budget]
    result = linprog(objective, A_ub=constraints, b_ub=rhs, A_eq=[np.r_[np.ones(n), np.zeros(n)]], b_eq=[1.0], bounds=(0,None), method="highs")
    assert result.success
    value, weights = robustness.worst_reweighting(values, budget)
    assert value == pytest.approx(result.fun, abs=1e-12)
    assert np.all(weights >= -1e-15)
    assert weights.sum() == pytest.approx(1.0)
    assert np.abs(weights - 1/n).sum()/2 <= budget + 1e-12
