import json
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
