from __future__ import annotations

from pathlib import Path
import unittest

from neurogenesis.policies import AdamLinearPolicy, FrozenLinearPolicy, SGDLinearPolicy
from neurogenesis.protocol import Protocol
from neurogenesis.runner import execute_task, load_development_config
from neurogenesis.tasks import TaskDefinition


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_development_config(ROOT / "configs" / "development.json")
        cls.definition = TaskDefinition.from_mapping(config["tasks"][0])
        cls.budget = Protocol.load(ROOT / "configs" / "protocol.json").run_budget

    def test_prediction_precedes_current_label_and_delay_is_enforced(self) -> None:
        policy = SGDLinearPolicy(feature_dim=8, learning_rate=0.01)
        metrics, trace = execute_task(self.definition, policy, self.budget)
        delay = self.definition.label_delay
        self.assertTrue(all(row["updates_before_prediction"] == 0 for row in trace[:delay]))
        self.assertEqual(trace[delay]["updates_before_prediction"], 1)
        self.assertEqual(metrics["resource_usage"]["parameter_updates"], 480 - delay)
        self.assertEqual(metrics["undelivered_labels_at_horizon"], delay)

    def test_no_adaptation_is_a_true_frozen_control(self) -> None:
        policy = FrozenLinearPolicy(feature_dim=8)
        metrics, _ = execute_task(self.definition, policy, self.budget)
        self.assertEqual(metrics["resource_usage"]["gradient_evaluations"], 0)
        self.assertEqual(metrics["resource_usage"]["parameter_updates"], 0)
        self.assertEqual(policy.weights, [0.0] * 8)

    def test_state_reset_is_exact_and_independent(self) -> None:
        first = AdamLinearPolicy(feature_dim=8, learning_rate=0.01)
        _, trace_first = execute_task(self.definition, first, self.budget)
        reset = AdamLinearPolicy(feature_dim=8, learning_rate=0.01)
        _, trace_reset = execute_task(self.definition, reset, self.budget)
        self.assertEqual(
            [row["prediction"] for row in trace_first],
            [row["prediction"] for row in trace_reset],
        )

    def test_shorter_horizon_and_larger_parameter_scale_are_supported(self) -> None:
        policy = SGDLinearPolicy(feature_dim=8, learning_rate=0.01)
        policy.weights = [5.0] * 8
        prediction = policy.predict([0.1] * 8)
        self.assertAlmostEqual(prediction, 4.0)
        self.assertEqual(policy.usage.parameter_updates, 0)


if __name__ == "__main__":
    unittest.main()

