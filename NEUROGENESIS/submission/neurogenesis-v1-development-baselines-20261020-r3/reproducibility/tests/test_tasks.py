from __future__ import annotations

from pathlib import Path
import unittest

from neurogenesis.runner import load_development_config
from neurogenesis.tasks import TaskDefinition, generate_task


ROOT = Path(__file__).resolve().parents[1]


class TaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_development_config(ROOT / "configs" / "development.json")
        cls.definition = TaskDefinition.from_mapping(config["tasks"][0])

    def test_generation_is_deterministic_and_contains_required_regimes(self) -> None:
        first = generate_task(self.definition)
        second = generate_task(self.definition)
        self.assertEqual(first, second)
        self.assertEqual(
            {sample.regime for sample in first},
            {"A_initial", "B_conflict", "A_return", "C_novel"},
        )

    def test_labels_are_delayed_and_distractors_exist(self) -> None:
        samples = generate_task(self.definition)
        self.assertGreater(self.definition.feature_dim, self.definition.informative_dim)
        for sample in samples:
            self.assertEqual(
                sample.label_available_at - sample.step,
                self.definition.label_delay,
            )

    def test_returning_regime_reuses_target_mapping(self) -> None:
        samples = generate_task(self.definition)
        length = self.definition.regime_length
        self.assertEqual(samples[0].regime, "A_initial")
        self.assertEqual(samples[2 * length].regime, "A_return")
        self.assertEqual(samples[length].regime, "B_conflict")


if __name__ == "__main__":
    unittest.main()

