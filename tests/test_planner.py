import json
import tempfile
import unittest
from pathlib import Path

from atlas.cli import main
from atlas.planner import build_manifest


class PlannerTests(unittest.TestCase):
    def test_default_manifest_is_valid_and_complete(self):
        manifest = build_manifest("Does latent prediction improve robustness under distribution shift?")
        self.assertEqual(manifest.metadata["experiment_count"], 15)
        self.assertEqual(len(manifest.experiments), 15)
        self.assertEqual(len({e.name for e in manifest.experiments}), 15)
        self.assertEqual(manifest.primary_criterion.operator, ">=")
        self.assertGreater(len(manifest.evidence_requirements), 3)
        manifest.validate()

    def test_run_id_is_deterministic_for_same_question(self):
        q = "Does intervention-aware training improve time-series robustness?"
        self.assertEqual(build_manifest(q).run_id, build_manifest(q).run_id)

    def test_rejects_duplicate_seeds(self):
        with self.assertRaisesRegex(ValueError, "seeds must be unique"):
            build_manifest(
                "Does the method beat the baseline on a preregistered evaluation?",
                seeds=(1, 1),
            )

    def test_rejects_identical_treatment_and_control(self):
        with self.assertRaisesRegex(ValueError, "treatment and control must differ"):
            build_manifest(
                "Does the method beat the baseline on a preregistered evaluation?",
                treatment="same",
                control="same",
            )

    def test_cli_writes_json_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run.json"
            rc = main([
                "init-run",
                "--question",
                "Does calibration improve under controlled multimodal perturbations?",
                "--treatment",
                "calibrated model",
                "--control",
                "uncalibrated model",
                "--metric",
                "ece",
                "--minimum-improvement",
                "0.01",
                "--seeds",
                "1,2",
                "--perturbations",
                "clean,occlusion",
                "--output",
                str(output),
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["experiment_count"], 4)
            self.assertEqual(len(payload["experiments"]), 4)


if __name__ == "__main__":
    unittest.main()
