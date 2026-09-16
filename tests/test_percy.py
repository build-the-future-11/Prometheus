import json
import tempfile
import unittest
from pathlib import Path

from atlas.cli import main
from atlas.percy import SCHEMA_VERSION, build_percy_export, verify_export_completion
from atlas.planner import build_manifest


class PercyExportTests(unittest.TestCase):
    def _manifest(self):
        return build_manifest(
            "Does a calibrated model remain reliable under controlled visual shift?",
            treatment="calibrated",
            control="uncalibrated",
            metric="ece",
            seeds=(1, 2),
            perturbations=("clean", "occlusion"),
        ).to_dict()

    def _export(self, **overrides):
        options = {
            "working_directory": "/workspace/project",
            "command_template": (
                "python run.py --experiment {experiment_id} --seed {seed} "
                "--condition {perturbation} --output {artifact_dir}"
            ),
            "artifact_root": "/tmp/evidence",
            "status_file": "/tmp/task-statuses.json",
            "verification_command": (
                "python -m atlas.cli verify-percy --export tasks.json "
                "--statuses statuses.json --artifact-root evidence"
            ),
            "timeout_seconds": 120,
            "max_retries": 2,
            "retry_backoff_seconds": 7,
            "resources": {"class": "heavy", "slots": 1},
        }
        options.update(overrides)
        return build_percy_export(self._manifest(), **options)

    def test_ids_are_deterministic_and_parent_depends_on_every_experiment(self):
        first = self._export()
        second = self._export()
        self.assertEqual(first["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            [task["task_id"] for task in first["tasks"]],
            [task["task_id"] for task in second["tasks"]],
        )
        experiments = [task for task in first["tasks"] if task["kind"] == "experiment"]
        verifier = [task for task in first["tasks"] if task["kind"] == "verification"][0]
        self.assertEqual(set(verifier["dependencies"]), {task["task_id"] for task in experiments})
        self.assertTrue(verifier["verification"]["requires_all_dependencies_succeeded"])

    def test_retry_and_required_artifacts_propagate(self):
        payload = self._export()
        source_by_name = {item["name"]: item for item in self._manifest()["experiments"]}
        for task in payload["tasks"]:
            if task["kind"] != "experiment":
                continue
            experiment_id = task["metadata"]["atlas_experiment_id"]
            self.assertEqual(task["retry_policy"], {"max_retries": 2, "backoff_seconds": 7})
            self.assertEqual(task["expected_artifacts"], list(source_by_name[experiment_id]["required_artifacts"]))
            self.assertIn(experiment_id, task["command"])
            self.assertIn(task["artifact_directory"], task["command"])

    def test_verifier_fails_on_failed_task_and_missing_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._export(artifact_root=root)
            experiments = [task for task in payload["tasks"] if task["kind"] == "experiment"]
            statuses = {}
            for task in experiments:
                statuses[task["task_id"]] = "SUCCEEDED"
                exp_dir = root / task["metadata"]["atlas_experiment_id"]
                exp_dir.mkdir(parents=True)
                for artifact in task["expected_artifacts"]:
                    target = exp_dir / artifact
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("ok", encoding="utf-8")

            failed = experiments[0]
            statuses[failed["task_id"]] = "FAILED"
            missing = experiments[1]
            (root / missing["metadata"]["atlas_experiment_id"] / missing["expected_artifacts"][0]).unlink()

            ok, errors = verify_export_completion(payload, task_statuses=statuses, artifact_root=root)
            self.assertFalse(ok)
            self.assertTrue(any("expected SUCCEEDED" in error for error in errors))
            self.assertTrue(any("missing required artifact" in error for error in errors))

    def test_verifier_passes_only_when_all_tasks_succeeded_and_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._export(artifact_root=root)
            experiments = [task for task in payload["tasks"] if task["kind"] == "experiment"]
            statuses = {}
            for task in experiments:
                statuses[task["task_id"]] = "SUCCEEDED"
                exp_dir = root / task["metadata"]["atlas_experiment_id"]
                exp_dir.mkdir(parents=True)
                for artifact in task["expected_artifacts"]:
                    target = exp_dir / artifact
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("ok", encoding="utf-8")

            ok, errors = verify_export_completion(payload, task_statuses=statuses, artifact_root=root)
            self.assertTrue(ok)
            self.assertEqual(errors, [])

    def test_cli_exports_without_percy_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.json"
            output_path = tmp_path / "percy-tasks.json"
            manifest_path.write_text(json.dumps(self._manifest()), encoding="utf-8")

            rc = main(
                [
                    "export-percy",
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--working-directory",
                    "/workspace/project",
                    "--command-template",
                    "python run.py --experiment {experiment_id} --output {artifact_dir}",
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_task_count"], 4)
            self.assertEqual(payload["task_count"], 5)
            verifier = [task for task in payload["tasks"] if task["kind"] == "verification"][0]
            self.assertIn("verify-percy", verifier["command"])


if __name__ == "__main__":
    unittest.main()
