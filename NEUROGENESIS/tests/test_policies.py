from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from neurogenesis.policies import (
    AdamLinearPolicy,
    SGDLinearPolicy,
    load_checkpoint,
    save_checkpoint,
    squared_error_gradient,
)


class PolicyTests(unittest.TestCase):
    def test_squared_error_gradient_matches_finite_difference(self) -> None:
        features = [0.7, -1.2, 0.3]
        weights = [0.2, -0.4, 0.1]
        bias = -0.05
        target = 0.8
        prediction = sum(w * x for w, x in zip(weights, features)) + bias
        analytic, analytic_bias = squared_error_gradient(features, target, prediction)
        epsilon = 1e-6

        def loss(test_weights: list[float], test_bias: float) -> float:
            estimate = sum(w * x for w, x in zip(test_weights, features)) + test_bias
            return 0.5 * (estimate - target) ** 2

        for index, expected in enumerate(analytic):
            plus = list(weights)
            minus = list(weights)
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical = (loss(plus, bias) - loss(minus, bias)) / (2 * epsilon)
            self.assertAlmostEqual(expected, numerical, places=7)
        numerical_bias = (loss(weights, bias + epsilon) - loss(weights, bias - epsilon)) / (
            2 * epsilon
        )
        self.assertAlmostEqual(analytic_bias, numerical_bias, places=7)

    def test_checkpoint_restoration_preserves_prediction_and_adam_state(self) -> None:
        policy = AdamLinearPolicy(feature_dim=3, learning_rate=0.01)
        features = [0.2, -0.5, 1.1]
        prediction = policy.predict(features)
        policy.observe_available_label(features, 0.7, prediction)
        expected = policy.predict(features)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_checkpoint(policy, path)
            restored = AdamLinearPolicy(feature_dim=3, learning_rate=0.01)
            load_checkpoint(restored, path)
        self.assertEqual(policy.snapshot(), restored.snapshot())
        self.assertEqual(expected, restored.predict(features))

    def test_large_features_remain_finite_under_gradient_clipping(self) -> None:
        policy = SGDLinearPolicy(feature_dim=2, learning_rate=0.05, gradient_clip=1.0)
        for _ in range(100):
            features = [1e6, -1e6]
            prediction = policy.predict(features)
            policy.observe_available_label(features, 1e6, prediction)
        self.assertTrue(all(math.isfinite(value) for value in policy.weights))
        self.assertTrue(math.isfinite(policy.bias))
        self.assertLessEqual(policy.max_update_norm, 0.05 + 1e-12)


if __name__ == "__main__":
    unittest.main()

