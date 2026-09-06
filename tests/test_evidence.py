import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from atlas.evidence import verify_evidence
from atlas.planner import build_manifest


class EvidenceTests(unittest.TestCase):
    def _manifest(self):
        return build_manifest(
            "Does the treatment improve the preregistered metric under clean evaluation?",
            seeds=(1,),
            perturbations=("clean",),
        )

    def test_reports_missing_artifacts(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            report = verify_evidence(manifest, tmp)
            self.assertEqual(report.status, "INCOMPLETE")
            self.assertEqual(report.expected_artifacts, 4)
            self.assertEqual(report.verified_artifacts, 0)
            self.assertEqual(len(report.missing), 4)

    def test_hashes_complete_evidence(self):
        manifest = self._manifest()
        experiment = manifest.experiments[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / experiment.name
            root.mkdir(parents=True)
            (root / "config.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
            (root / "metrics.json").write_text(json.dumps({"score": 0.5}), encoding="utf-8")
            (root / "stdout.log").write_text("ok\n", encoding="utf-8")
            (root / "environment.txt").write_text("python=3.11\n", encoding="utf-8")

            report = verify_evidence(manifest, tmp)
            self.assertEqual(report.status, "COMPLETE")
            self.assertEqual(report.verified_artifacts, 4)
            self.assertEqual(report.missing, [])
            self.assertEqual(report.invalid_json, [])
            self.assertTrue(all(len(record.sha256) == 64 for record in report.artifacts))

    def test_rejects_non_object_json_artifacts(self):
        manifest = self._manifest()
        experiment = manifest.experiments[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / experiment.name
            root.mkdir(parents=True)
            (root / "config.json").write_text("[]", encoding="utf-8")
            (root / "metrics.json").write_text(json.dumps({"score": 0.5}), encoding="utf-8")
            (root / "stdout.log").write_text("ok\n", encoding="utf-8")
            (root / "environment.txt").write_text("python=3.11\n", encoding="utf-8")

            report = verify_evidence(manifest, tmp)
            self.assertEqual(report.status, "INCOMPLETE")
            self.assertEqual(report.invalid_json, [f"{experiment.name}/config.json"])

    def test_rejects_artifact_traversal_before_reading_outside_root(self):
        manifest = self._manifest()
        experiment = manifest.experiments[0]
        manifest.experiments = [
            replace(experiment, required_artifacts=("../outside.json",))
        ]

        with tempfile.TemporaryDirectory() as tmp:
            evidence_root = Path(tmp) / "evidence"
            (evidence_root / experiment.name).mkdir(parents=True)
            (evidence_root / "outside.json").write_text(
                json.dumps({"must_not": "be accepted"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "safe relative artifact path"):
                verify_evidence(manifest, evidence_root)

    def test_rejects_unsafe_experiment_directory_name(self):
        manifest = self._manifest()
        experiment = manifest.experiments[0]
        manifest.experiments = [replace(experiment, name="../escape")]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "single safe path component"):
                verify_evidence(manifest, tmp)

    def test_rejects_duplicate_required_artifact_contract(self):
        manifest = self._manifest()
        experiment = manifest.experiments[0]
        manifest.experiments = [
            replace(experiment, required_artifacts=("config.json", "config.json"))
        ]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "duplicate required artifact"):
                verify_evidence(manifest, tmp)

    def test_rejects_symlinked_artifact_that_escapes_evidence_root(self):
        manifest = self._manifest()
        experiment = manifest.experiments[0]
        manifest.experiments = [replace(experiment, required_artifacts=("config.json",))]

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            evidence_root = base / "evidence"
            experiment_root = evidence_root / experiment.name
            experiment_root.mkdir(parents=True)
            outside = base / "outside.json"
            outside.write_text(json.dumps({"outside": True}), encoding="utf-8")
            try:
                (experiment_root / "config.json").symlink_to(outside)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "escapes evidence root"):
                verify_evidence(manifest, evidence_root)


if __name__ == "__main__":
    unittest.main()
