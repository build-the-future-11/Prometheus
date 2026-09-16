import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from atlas.cli import main
from atlas.percy import build_percy_export, write_percy_export


class PercyCliPathContractTests(unittest.TestCase):
    def _manifest(self):
        return {
            "run_id": "synthetic-path-contract",
            "experiments": [
                {
                    "name": "clean",
                    "seed": 0,
                    "perturbation": "fixture",
                    "treatment": "fixture",
                    "control": "fixture",
                    "required_artifacts": ["metrics.json"],
                }
            ],
        }

    def _materialized_export(self, root: Path):
        export_path = root / "tasks.json"
        artifact_root = root / "evidence"
        status_path = root / "statuses.json"
        payload = build_percy_export(
            self._manifest(),
            working_directory="/workspace/project",
            command_template="python fixture_only.py --output {artifact_dir}",
            artifact_root=artifact_root,
            status_file=status_path,
            verification_command=(
                f"python -m atlas.cli verify-percy --export {export_path} "
                f"--statuses {status_path} --artifact-root {artifact_root}"
            ),
        )
        write_percy_export(payload, export_path)

        statuses = {}
        for task in payload["tasks"]:
            if task["kind"] != "experiment":
                continue
            statuses[task["task_id"]] = "SUCCEEDED"
            target = artifact_root / task["metadata"]["atlas_experiment_id"] / "metrics.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('{"synthetic_fixture": true}\n', encoding="utf-8")
        status_path.write_text(json.dumps(statuses), encoding="utf-8")
        return export_path, artifact_root, status_path, payload, statuses

    def _verify(self, export_path: Path, status_path: Path, artifact_root: Path) -> int:
        with redirect_stdout(io.StringIO()):
            return main(
                [
                    "verify-percy",
                    "--export",
                    str(export_path),
                    "--statuses",
                    str(status_path),
                    "--artifact-root",
                    str(artifact_root),
                ]
            )

    def test_cli_export_normalizes_executor_paths_to_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(self._manifest()), encoding="utf-8")
            previous_cwd = Path.cwd()
            os.chdir(root)
            try:
                with redirect_stdout(io.StringIO()):
                    rc = main(
                        [
                            "export-percy",
                            "--manifest",
                            "manifest.json",
                            "--output",
                            "nested/tasks.json",
                            "--working-directory",
                            "/workspace/project",
                            "--command-template",
                            "python fixture_only.py --output {artifact_dir}",
                            "--artifact-root",
                            "relative-artifacts",
                            "--status-file",
                            "relative-statuses.json",
                        ]
                    )
                self.assertEqual(rc, 0)
                export_path = (root / "nested/tasks.json").resolve()
                payload = json.loads(export_path.read_text(encoding="utf-8"))
                expected_artifact_root = (root / "relative-artifacts").resolve()
                expected_status_path = (root / "relative-statuses.json").resolve()
                self.assertEqual(Path(payload["status_contract"]["path"]), expected_status_path)
                verifier = next(task for task in payload["tasks"] if task["kind"] == "verification")
                self.assertEqual(Path(verifier["artifact_directory"]), expected_artifact_root)
                experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
                self.assertEqual(
                    Path(experiment["artifact_directory"]),
                    expected_artifact_root / experiment["metadata"]["atlas_experiment_id"],
                )
                self.assertIn(str(export_path), verifier["command"])
                self.assertIn(str(expected_status_path), verifier["command"])
                self.assertIn(str(expected_artifact_root), verifier["command"])
            finally:
                os.chdir(previous_cwd)

    def test_cli_verification_accepts_only_declared_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_path, artifact_root, status_path, _, _ = self._materialized_export(root)
            self.assertEqual(self._verify(export_path, status_path, artifact_root), 0)

    def test_cli_verification_rejects_substitute_status_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_path, artifact_root, status_path, _, statuses = self._materialized_export(root)
            substitute = root / "substitute-statuses.json"
            substitute.write_text(json.dumps(statuses), encoding="utf-8")
            self.assertEqual(self._verify(export_path, substitute, artifact_root), 2)
            self.assertTrue(status_path.exists())

    def test_cli_verification_rejects_substitute_artifact_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_path, artifact_root, status_path, payload, _ = self._materialized_export(root)
            substitute = root / "substitute-evidence"
            experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
            target = substitute / experiment["metadata"]["atlas_experiment_id"] / "metrics.json"
            target.parent.mkdir(parents=True)
            target.write_text('{"synthetic_fixture": true}\n', encoding="utf-8")
            self.assertEqual(self._verify(export_path, status_path, substitute), 2)
            self.assertTrue(artifact_root.exists())

    def test_cli_verification_rejects_task_artifact_directory_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_path, artifact_root, status_path, payload, _ = self._materialized_export(root)
            experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
            experiment["artifact_directory"] = str(root / "other" / experiment["metadata"]["atlas_experiment_id"])
            write_percy_export(payload, export_path)
            self.assertEqual(self._verify(export_path, status_path, artifact_root), 2)


if __name__ == "__main__":
    unittest.main()
