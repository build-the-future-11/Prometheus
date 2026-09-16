import tempfile
import unittest
from pathlib import Path

from atlas.percy import build_percy_export, verify_export_completion
from atlas.planner import build_manifest


class PercyBoundaryTests(unittest.TestCase):
    def _manifest(self):
        return build_manifest(
            "Does a calibrated model remain reliable under controlled visual shift?",
            treatment="calibrated",
            control="uncalibrated",
            metric="ece",
            seeds=(1,),
            perturbations=("clean",),
        ).to_dict()

    def _export(self, manifest=None, **overrides):
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
        }
        options.update(overrides)
        return build_percy_export(manifest or self._manifest(), **options)

    def _materialize_success(self, payload, root):
        statuses = {}
        for task in payload["tasks"]:
            if task["kind"] != "experiment":
                continue
            statuses[task["task_id"]] = "SUCCEEDED"
            exp_dir = root / task["metadata"]["atlas_experiment_id"]
            exp_dir.mkdir(parents=True)
            for artifact in task["expected_artifacts"]:
                target = exp_dir / artifact
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("ok", encoding="utf-8")
        return statuses

    def test_export_rejects_experiment_directory_traversal(self):
        manifest = self._manifest()
        manifest["experiments"][0]["name"] = "../outside"
        with self.assertRaisesRegex(ValueError, "single safe path component"):
            self._export(manifest)

        manifest = self._manifest()
        manifest["experiments"][0]["name"] = r"..\outside"
        with self.assertRaisesRegex(ValueError, "single safe path component"):
            self._export(manifest)

    def test_export_rejects_cross_platform_artifact_traversal(self):
        for artifact in ("../secret.txt", r"..\secret.txt", "/tmp/secret.txt", r"C:\secret.txt"):
            with self.subTest(artifact=artifact):
                manifest = self._manifest()
                manifest["experiments"][0]["required_artifacts"] = [artifact]
                with self.assertRaisesRegex(ValueError, "safe relative artifact path"):
                    self._export(manifest)

    def test_verifier_rejects_tampered_parent_contract(self):
        payload = self._export()
        verifier = next(task for task in payload["tasks"] if task["kind"] == "verification")
        verifier["dependencies"] = []

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "dependencies do not exactly match"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_expected_artifact_contract_drift(self):
        payload = self._export()
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        experiment["expected_artifacts"] = ["metrics.json"]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "artifact contract drifted"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_experiment_id_traversal_even_if_payload_is_tampered(self):
        payload = self._export()
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        experiment["metadata"]["atlas_experiment_id"] = "../outside"

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "single safe path component"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_rewritten_experiment_task_identity_even_if_contract_matches(self):
        payload = self._export()
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        verifier = next(task for task in payload["tasks"] if task["kind"] == "verification")
        original_id = experiment["task_id"]
        forged_id = "atlas-experiment-0000000000000000"
        experiment["task_id"] = forged_id
        verifier["dependencies"] = [
            forged_id if task_id == original_id else task_id for task_id in verifier["dependencies"]
        ]
        required_by_task = verifier["verification"]["required_artifacts_by_task"]
        required_by_task[forged_id] = required_by_task.pop(original_id)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            statuses = self._materialize_success(payload, root)
            with self.assertRaisesRegex(ValueError, "deterministic stable identity"):
                verify_export_completion(payload, task_statuses=statuses, artifact_root=root)

    def test_verifier_rejects_rewritten_verification_task_identity(self):
        payload = self._export()
        verifier = next(task for task in payload["tasks"] if task["kind"] == "verification")
        forged_id = "atlas-verify-0000000000000000"
        verifier["task_id"] = forged_id
        payload["verification_task_id"] = forged_id

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "deterministic stable identity"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_cross_run_metadata_identity(self):
        payload = self._export()
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        experiment["metadata"]["atlas_run_id"] = "different-run"

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "metadata.atlas_run_id does not match"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_parent_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            payload = self._export(artifact_root=root)
            outside = Path(outside_tmp)
            experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
            experiment_id = experiment["metadata"]["atlas_experiment_id"]
            outside_exp = outside / experiment_id
            outside_exp.mkdir()
            for artifact in experiment["expected_artifacts"]:
                target = outside_exp / artifact
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("outside", encoding="utf-8")
            (root / experiment_id).symlink_to(outside_exp, target_is_directory=True)
            statuses = {experiment["task_id"]: "SUCCEEDED"}

            ok, errors = verify_export_completion(payload, task_statuses=statuses, artifact_root=root)

            self.assertFalse(ok)
            self.assertTrue(any("artifact escapes artifact root" in error for error in errors))

    def test_verifier_accepts_regular_files_inside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._export(artifact_root=root)
            statuses = self._materialize_success(payload, root)

            ok, errors = verify_export_completion(payload, task_statuses=statuses, artifact_root=root)

            self.assertTrue(ok)
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
