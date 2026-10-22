import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from derived_learning.audit import audit_global_ridge, build_claim_status
from derived_learning.experiment import execute_experiment, load_config, validate_experiment_config
from derived_learning.report import render_report
from derived_learning.verification import verify_manifest, verify_saved_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def test_smoke_execution_has_locked_splits_and_verified_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            manifest = execute_experiment(PROJECT_ROOT / "configs" / "smoke.json", run_dir, PROJECT_ROOT)
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertFalse(json.loads((run_dir / "selection.json").read_text())["uses_final_split"])
            selection_text = (run_dir / "selection_attempts.jsonl").read_text()
            self.assertNotIn('"split":"final"', selection_text)
            results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(results), 8)
            self.assertTrue(all(result["status"] == "SUCCEEDED" for result in results))
            for relative, expected in manifest["outputs"].items():
                actual = hashlib.sha256((run_dir / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)
            self.assertEqual(verify_manifest(run_dir)["status"], "VERIFIED")
            evaluation = verify_saved_evaluation(run_dir)
            self.assertEqual(evaluation["status"], "VERIFIED")
            self.assertEqual(evaluation["checked_result_count"], 8)
            report = render_report(run_dir)
            self.assertIn("Equal-resource mean NMSE", report)
            self.assertIn("internally executed synthetic study", report)
            audit_path = run_dir / "global_ridge_audit.json"
            audit = audit_global_ridge(run_dir, PROJECT_ROOT)
            audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(audit["status"], "VERIFIED")
            self.assertTrue(audit["records"])
            claims = build_claim_status(run_dir, audit_path)
            self.assertFalse(claims["broad_superiority_claim_allowed"])
            self.assertFalse(claims["submission_ready"])

    def test_relative_config_path_is_recorded_relative_to_project(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(PROJECT_ROOT)
                manifest = execute_experiment(Path("configs/smoke.json"), Path(directory) / "run", Path("."))
            finally:
                os.chdir(previous)
            self.assertEqual(manifest["config_path"], "configs/smoke.json")

    def test_config_validator_rejects_cross_split_seed_reuse(self):
        config = load_config(PROJECT_ROOT / "configs" / "smoke.json")
        reused = json.loads(json.dumps(config))
        reused["splits"]["development"]["seeds"][0] = reused["splits"]["meta_train"]["seeds"][0]
        with self.assertRaisesRegex(ValueError, "reused"):
            validate_experiment_config(reused)


if __name__ == "__main__":
    unittest.main()

