import math
import unittest

from atlas.models import Criterion, Experiment


class CriterionValidationTests(unittest.TestCase):
    def test_rejects_non_finite_thresholds(self):
        for threshold in (math.nan, math.inf, -math.inf):
            with self.subTest(threshold=threshold):
                criterion = Criterion(
                    metric="primary_metric",
                    operator=">=",
                    threshold=threshold,
                    rationale="Frozen primary success threshold.",
                )
                with self.assertRaisesRegex(ValueError, "threshold must be finite"):
                    criterion.validate()

    def test_accepts_finite_threshold(self):
        Criterion(
            metric="primary_metric",
            operator=">=",
            threshold=0.0,
            rationale="Frozen primary success threshold.",
        ).validate()


class ExperimentValidationTests(unittest.TestCase):
    def _experiment(self, **overrides):
        values = {
            "name": "shift-seed-17",
            "treatment": "proposed method",
            "control": "strong baseline",
            "seed": 17,
            "perturbation": "distribution_shift",
            "required_artifacts": ("config.json", "metrics.json"),
        }
        values.update(overrides)
        return Experiment(**values)

    def test_rejects_blank_treatment_or_control(self):
        for field in ("treatment", "control"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "treatment and control must be non-empty"):
                    self._experiment(**{field: "   "}).validate()

    def test_rejects_blank_perturbation(self):
        with self.assertRaisesRegex(ValueError, "perturbation must be non-empty"):
            self._experiment(perturbation="  ").validate()

    def test_rejects_blank_required_artifact(self):
        with self.assertRaisesRegex(ValueError, "required artifact paths must be non-empty strings"):
            self._experiment(required_artifacts=("config.json", "   ")).validate()

    def test_rejects_duplicate_required_artifact(self):
        with self.assertRaisesRegex(ValueError, "required artifact paths must be unique"):
            self._experiment(required_artifacts=("metrics.json", "metrics.json")).validate()

    def test_accepts_valid_experiment_contract(self):
        self._experiment(required_artifacts=("config.json", "nested/metrics.json")).validate()


if __name__ == "__main__":
    unittest.main()
