import math
import unittest

from atlas.models import Criterion, Experiment, RunManifest


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

    def test_rejects_boolean_or_non_integer_seed(self):
        for seed in (True, False, 1.5, "17"):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seed must be a non-negative integer"):
                    self._experiment(seed=seed).validate()

    def test_rejects_blank_required_artifact(self):
        with self.assertRaisesRegex(ValueError, "required artifact paths must be non-empty strings"):
            self._experiment(required_artifacts=("config.json", "   ")).validate()

    def test_rejects_duplicate_required_artifact(self):
        with self.assertRaisesRegex(ValueError, "required artifact paths must be unique"):
            self._experiment(required_artifacts=("metrics.json", "metrics.json")).validate()

    def test_accepts_valid_experiment_contract(self):
        self._experiment(required_artifacts=("config.json", "nested/metrics.json")).validate()


class RunManifestValidationTests(unittest.TestCase):
    def _manifest(self, **overrides):
        values = {
            "run_id": "atlas-run-001",
            "question": "Does the treatment improve the frozen primary metric?",
            "hypothesis": "Treatment improves the frozen primary metric.",
            "null_hypothesis": "Treatment does not improve the frozen primary metric.",
            "primary_criterion": Criterion(
                metric="primary_metric",
                operator=">=",
                threshold=0.5,
                rationale="Frozen before execution.",
            ),
            "experiments": [
                Experiment(
                    name="shift-seed-17",
                    treatment="proposed method",
                    control="strong baseline",
                    seed=17,
                    perturbation="distribution_shift",
                    required_artifacts=("config.json", "metrics.json"),
                )
            ],
            "stopping_rule": "Run the frozen experiment exactly once.",
            "evidence_requirements": ["config.json", "metrics.json"],
        }
        values.update(overrides)
        return RunManifest(**values)

    def test_rejects_blank_or_non_string_run_id(self):
        for run_id in ("", "   ", None, 17):
            with self.subTest(run_id=run_id):
                with self.assertRaisesRegex(ValueError, "run_id must be a non-empty string"):
                    self._manifest(run_id=run_id).validate()

    def test_rejects_blank_or_non_string_evidence_requirement(self):
        for requirement in ("", "   ", None, 17):
            with self.subTest(requirement=requirement):
                with self.assertRaisesRegex(ValueError, "evidence requirements must be non-empty strings"):
                    self._manifest(evidence_requirements=["config.json", requirement]).validate()

    def test_rejects_duplicate_evidence_requirements(self):
        with self.assertRaisesRegex(ValueError, "evidence requirements must be unique"):
            self._manifest(evidence_requirements=["metrics.json", "metrics.json"]).validate()

    def test_accepts_valid_manifest_contract(self):
        manifest = self._manifest()
        manifest.validate()
        self.assertEqual(manifest.to_dict()["run_id"], "atlas-run-001")


if __name__ == "__main__":
    unittest.main()
