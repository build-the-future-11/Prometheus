import tempfile
import unittest

from atlas.percy import build_percy_export, verify_export_completion
from atlas.planner import build_manifest


class PercyExportStructureContractTests(unittest.TestCase):
    def _export(self):
        manifest = build_manifest(
            "Does a calibrated model remain reliable under controlled visual shift?",
            treatment="calibrated",
            control="uncalibrated",
            metric="ece",
            seeds=(1,),
            perturbations=("clean",),
        ).to_dict()
        return build_percy_export(
            manifest,
            working_directory="/workspace/project",
            command_template="python run.py --experiment {experiment_id} --output {artifact_dir}",
            artifact_root="/tmp/evidence",
            status_file="/tmp/task-statuses.json",
            verification_command="python -m atlas.cli verify-percy",
        )

    def test_verifier_rejects_experiment_dependency_topology_drift(self):
        payload = self._export()
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        experiment["dependencies"] = [payload["verification_task_id"]]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "experiment dependencies must be empty"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_missing_status_contract(self):
        payload = self._export()
        payload.pop("status_contract")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "status_contract is malformed"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_success_status_contract_drift(self):
        payload = self._export()
        payload["status_contract"]["success_status"] = "DONE"

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "success_status must be SUCCEEDED"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_relative_serialized_status_path(self):
        payload = self._export()
        payload["status_contract"]["path"] = "task-statuses.json"

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "export.status_contract.path must be an absolute path"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_relative_verification_working_directory(self):
        payload = self._export()
        verifier = next(task for task in payload["tasks"] if task["kind"] == "verification")
        verifier["working_directory"] = "workspace/project"

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "verification.working_directory must be an absolute path"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_experiment_working_directory_drift(self):
        payload = self._export()
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        experiment["working_directory"] = "/workspace/other"

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "working_directory drifted from verification task"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_verifier_rejects_experiment_artifact_directory_drift(self):
        payload = self._export()
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        experiment["artifact_directory"] = "/tmp/other"

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "artifact_directory drifted from verification root"):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)

    def test_library_verifier_rejects_substitute_runtime_artifact_root(self):
        payload = self._export()

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError, "artifact_root does not match verification artifact_directory"
            ):
                verify_export_completion(payload, task_statuses={}, artifact_root=tmp)


if __name__ == "__main__":
    unittest.main()
