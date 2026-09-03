import json
import tempfile
import unittest
from pathlib import Path

from atlas.claims import ClaimSpec, classify_claim
from atlas.planner import build_manifest


class ClaimTests(unittest.TestCase):
    def _manifest(self):
        return build_manifest(
            "Does the treatment improve robustness across preregistered seeds?",
            seeds=(1, 2),
            perturbations=("shift",),
        )

    def _write_experiment(self, root: Path, name: str, metric_payload):
        exp = root / name
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "config.json").write_text(json.dumps({"experiment": name}), encoding="utf-8")
        (exp / "metrics.json").write_text(json.dumps(metric_payload), encoding="utf-8")
        (exp / "stdout.log").write_text("complete\n", encoding="utf-8")
        (exp / "environment.txt").write_text("python=3.11\n", encoding="utf-8")

    def test_supported_claim(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": 0.04})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": 0.06})
            report = classify_claim(manifest, root, ClaimSpec("improvement", ">=", 0.05))
            self.assertEqual(report.classification, "SUPPORTED")
            self.assertEqual(report.evidence_status, "COMPLETE")
            self.assertAlmostEqual(report.aggregate_value, 0.05)
            self.assertEqual(len(report.values_by_experiment), 2)

    def test_not_supported_claim(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": 0.01})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": 0.02})
            report = classify_claim(manifest, root, ClaimSpec("improvement", ">=", 0.05))
            self.assertEqual(report.classification, "NOT_SUPPORTED")
            self.assertAlmostEqual(report.aggregate_value, 0.015)

    def test_missing_preregistered_run_is_inconclusive(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": 0.99})
            report = classify_claim(manifest, root, ClaimSpec("improvement", ">=", 0.05))
            self.assertEqual(report.classification, "INCONCLUSIVE")
            self.assertEqual(report.evidence_status, "INCOMPLETE")
            self.assertEqual(len(report.missing_metric_experiments), 1)

    def test_malformed_metric_is_invalid_evidence(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": "high"})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": 0.06})
            report = classify_claim(manifest, root, ClaimSpec("improvement", ">=", 0.05))
            self.assertEqual(report.classification, "INVALID_EVIDENCE")
            self.assertEqual(report.malformed_metric_experiments, [manifest.experiments[0].name])

    def test_mixed_seed_values_are_preserved(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": 1.0})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": -1.0})
            report = classify_claim(manifest, root, ClaimSpec("improvement", ">", 0.1))
            self.assertEqual(report.classification, "NOT_SUPPORTED")
            self.assertEqual(set(report.values_by_experiment.values()), {1.0, -1.0})
            self.assertAlmostEqual(report.aggregate_value, 0.0)


if __name__ == "__main__":
    unittest.main()
