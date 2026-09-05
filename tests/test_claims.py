import json
import tempfile
import unittest
from pathlib import Path

from atlas.claims import ClaimSpec, classify_claim, load_claim_spec
from atlas.planner import build_manifest


class ClaimTests(unittest.TestCase):
    def _manifest(self):
        return build_manifest(
            "Does the treatment improve robustness across preregistered seeds?",
            seeds=(1, 2),
            perturbations=("shift",),
        )

    def _spec(self, manifest, **overrides):
        params = {
            "metric_key": "improvement",
            "operator": ">=",
            "threshold": 0.05,
            "tolerance": 0.0,
            "aggregation_rule": "mean",
            "experiment_ids": tuple(exp.name for exp in manifest.experiments),
        }
        params.update(overrides)
        return ClaimSpec(**params)

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
            report = classify_claim(manifest, root, self._spec(manifest))
            self.assertEqual(report.classification, "SUPPORTED")
            self.assertEqual(report.evidence_status, "COMPLETE")
            self.assertAlmostEqual(report.aggregate_value, 0.05)
            self.assertEqual(len(report.values_by_experiment), 2)
            self.assertEqual(report.aggregation_rule, "mean")

    def test_not_supported_claim(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": 0.01})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": 0.02})
            report = classify_claim(manifest, root, self._spec(manifest))
            self.assertEqual(report.classification, "NOT_SUPPORTED")
            self.assertAlmostEqual(report.aggregate_value, 0.015)

    def test_missing_preregistered_run_is_inconclusive(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": 0.99})
            report = classify_claim(manifest, root, self._spec(manifest))
            self.assertEqual(report.classification, "INCONCLUSIVE")
            self.assertEqual(report.evidence_status, "INCOMPLETE")
            self.assertEqual(len(report.missing_metric_experiments), 1)

    def test_explicit_failed_run_is_inconclusive_even_with_good_metric(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"status": "FAILED", "improvement": 1.0})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": 1.0})
            report = classify_claim(manifest, root, self._spec(manifest))
            self.assertEqual(report.classification, "INCONCLUSIVE")
            self.assertEqual(report.failed_experiments, [manifest.experiments[0].name])

    def test_malformed_metric_is_invalid_evidence(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": "high"})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": 0.06})
            report = classify_claim(manifest, root, self._spec(manifest))
            self.assertEqual(report.classification, "INVALID_EVIDENCE")
            self.assertEqual(report.malformed_metric_experiments, [manifest.experiments[0].name])

    def test_mixed_seed_values_are_preserved(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": 1.0})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": -1.0})
            report = classify_claim(
                manifest,
                root,
                self._spec(manifest, operator=">", threshold=0.1),
            )
            self.assertEqual(report.classification, "NOT_SUPPORTED")
            self.assertEqual(set(report.values_by_experiment.values()), {1.0, -1.0})
            self.assertAlmostEqual(report.aggregate_value, 0.0)

    def test_aggregation_rule_is_frozen_and_applied(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": 0.01})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": 0.09})
            report = classify_claim(
                manifest,
                root,
                self._spec(manifest, aggregation_rule="min", threshold=0.05),
            )
            self.assertEqual(report.classification, "NOT_SUPPORTED")
            self.assertAlmostEqual(report.aggregate_value, 0.01)

    def test_tolerance_is_explicit_and_applied(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, manifest.experiments[0].name, {"improvement": 0.048})
            self._write_experiment(root, manifest.experiments[1].name, {"improvement": 0.048})
            strict = classify_claim(manifest, root, self._spec(manifest, tolerance=0.0))
            tolerant = classify_claim(manifest, root, self._spec(manifest, tolerance=0.002))
            self.assertEqual(strict.classification, "NOT_SUPPORTED")
            self.assertEqual(tolerant.classification, "SUPPORTED")

    def test_spec_roundtrip_is_machine_readable(self):
        manifest = self._manifest()
        spec = self._spec(manifest, aggregation_rule="median", tolerance=0.001)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claim.json"
            path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
            loaded = load_claim_spec(path)
            self.assertEqual(loaded, spec)
            self.assertEqual(loaded.to_dict()["direction"], "higher")

    def test_spec_cannot_omit_a_preregistered_experiment(self):
        manifest = self._manifest()
        spec = self._spec(manifest, experiment_ids=(manifest.experiments[0].name,))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "exactly freeze the manifest experiment set"):
                classify_claim(manifest, root, spec)


if __name__ == "__main__":
    unittest.main()
