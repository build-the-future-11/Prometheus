import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from derived_learning.linalg import mse
from derived_learning.model import GroupedLinearModel
from derived_learning.rules import (
    AdamRule,
    AnalyticResidualRule,
    BlockGradientAblationRule,
    BlockJacobiResidualRule,
    ClassicalBackfittingRule,
    FixedNormalizedLMSRule,
    GlobalRidgeRule,
    ResourceCounter,
    mse_gradient,
)


class ModelAndRuleTests(unittest.TestCase):
    def setUp(self):
        self.rows = [[1.0, 0.0, 0.5, 1.0], [0.0, 1.0, 1.0, -0.5], [1.0, 1.0, -1.0, 0.25]]
        self.targets = [2.0, -1.0, 0.5]
        self.groups = [[0, 1], [2, 3]]

    def test_analytic_gradient_matches_finite_difference(self):
        model = GroupedLinearModel(4, self.groups, [0.2, -0.1, 0.4, 0.3])
        analytic = mse_gradient(model, self.rows, self.targets)
        epsilon = 1e-6
        for index in range(model.dimension):
            original = model.weights[index]
            model.weights[index] = original + epsilon
            plus = mse(model.predict(self.rows), self.targets)
            model.weights[index] = original - epsilon
            minus = mse(model.predict(self.rows), self.targets)
            model.weights[index] = original
            numeric = (plus - minus) / (2.0 * epsilon)
            self.assertAlmostEqual(analytic[index], numeric, places=6)

    def test_derived_rule_does_not_call_global_gradient(self):
        model = GroupedLinearModel.zeros(4, self.groups)
        resources = ResourceCounter()
        with patch("derived_learning.rules.mse_gradient", side_effect=AssertionError("global gradient called")):
            AnalyticResidualRule(regularization=0.1).step(model, self.rows, self.targets, resources)
        self.assertEqual(resources.full_global_gradient_calls, 0)
        self.assertGreater(resources.arithmetic_work_units, 0)

    def test_gradient_isolation_for_group_assignment(self):
        model = GroupedLinearModel(4, self.groups, [1.0, 2.0, 3.0, 4.0])
        model.set_group_weights([0, 1], [9.0, 8.0])
        self.assertEqual(model.weights, [9.0, 8.0, 3.0, 4.0])

    def test_affine_intercept_is_learned_without_changing_block_shape(self):
        rows = [[0.0, 0.0, 0.0, 0.0] for _ in range(3)]
        targets = [3.0, 3.0, 3.0]
        model = GroupedLinearModel.zeros(4, self.groups, fit_intercept=True)
        AnalyticResidualRule(0.1).step(model, rows, targets, ResourceCounter())
        self.assertAlmostEqual(model.intercept, 3.0)
        self.assertEqual(model.weights, [0.0, 0.0, 0.0, 0.0])

    def test_all_rules_accept_batches_and_remain_finite(self):
        constructors = [
            lambda: AdamRule(4, 0.01),
            lambda: AnalyticResidualRule(0.1),
            lambda: BlockJacobiResidualRule(0.1),
            lambda: FixedNormalizedLMSRule(),
            lambda: BlockGradientAblationRule(0.01),
            lambda: ClassicalBackfittingRule(0.1),
            lambda: GlobalRidgeRule(0.1),
        ]
        for constructor in constructors:
            with self.subTest(rule=constructor().__class__.__name__):
                model = GroupedLinearModel.zeros(4, self.groups)
                constructor().step(model, self.rows, self.targets, ResourceCounter())
                self.assertTrue(all(math.isfinite(value) for value in model.weights))

    def test_analytic_residual_is_exactly_classical_backfitting(self):
        analytic = GroupedLinearModel.zeros(4, self.groups)
        classical = GroupedLinearModel.zeros(4, self.groups)
        analytic_resources = ResourceCounter()
        classical_resources = ResourceCounter()
        AnalyticResidualRule(0.1).step(analytic, self.rows, self.targets, analytic_resources)
        ClassicalBackfittingRule(0.1).step(classical, self.rows, self.targets, classical_resources)
        self.assertEqual(analytic.to_dict(), classical.to_dict())
        self.assertEqual(analytic_resources.to_dict(), classical_resources.to_dict())

    def test_cyclic_backfitting_converges_to_global_ridge(self):
        analytic = GroupedLinearModel.zeros(4, self.groups)
        for _ in range(1_000):
            AnalyticResidualRule(0.1).step(analytic, self.rows, self.targets, ResourceCounter())
        optimum = GroupedLinearModel.zeros(4, self.groups)
        global_rule = GlobalRidgeRule(0.1)
        global_rule.step(optimum, self.rows, self.targets, ResourceCounter())
        self.assertTrue(global_rule.converged)
        for left, right in zip(analytic.weights, optimum.weights, strict=True):
            self.assertAlmostEqual(left, right, places=10)

    def test_bad_batch_shape_and_nonfinite_values_are_rejected(self):
        model = GroupedLinearModel.zeros(4, self.groups)
        with self.assertRaises(ValueError):
            AdamRule(4, 0.01).step(model, [[1.0]], [1.0], ResourceCounter())
        with self.assertRaises(ValueError):
            FixedNormalizedLMSRule().step(model, [[1.0, 2.0, 3.0, math.nan]], [1.0], ResourceCounter())

    def test_model_round_trip(self):
        model = GroupedLinearModel(4, self.groups, [1.0, 2.0, 3.0, 4.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            self.assertEqual(GroupedLinearModel.load(path).to_dict(), model.to_dict())

    def test_adam_state_round_trip_resumes_identically(self):
        first_model = GroupedLinearModel.zeros(4, self.groups)
        first_rule = AdamRule(4, 0.03)
        first_rule.step(first_model, self.rows, self.targets, ResourceCounter())
        state = json.loads(json.dumps(first_rule.state_dict()))
        resumed_model = GroupedLinearModel.from_dict(first_model.to_dict())
        resumed_rule = AdamRule.from_state_dict(state)
        first_rule.step(first_model, self.rows, self.targets, ResourceCounter())
        resumed_rule.step(resumed_model, self.rows, self.targets, ResourceCounter())
        self.assertEqual(first_model.weights, resumed_model.weights)
        self.assertEqual(first_rule.state_dict(), resumed_rule.state_dict())


if __name__ == "__main__":
    unittest.main()

