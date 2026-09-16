import copy
import unittest

from atlas.percy import build_percy_export, verify_export_completion


class PercySerializedContractDriftTests(unittest.TestCase):
    def _export(self):
        return build_percy_export(
            {
                "run_id": "serialized-contract-drift",
                "experiments": [
                    {
                        "name": "clean",
                        "seed": 0,
                        "perturbation": "fixture",
                        "treatment": "fixture-treatment",
                        "control": "fixture-control",
                        "required_artifacts": ["nested/metrics.json"],
                    }
                ],
            },
            working_directory="/workspace/project",
            command_template="python fixture_only.py --output {artifact_dir}",
            artifact_root="/tmp/evidence",
            status_file="/tmp/statuses.json",
            verification_command="python -m atlas.cli verify-percy",
        )

    def _experiment_and_verifier(self, payload):
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        verifier = next(task for task in payload["tasks"] if task["kind"] == "verification")
        return experiment, verifier

    def test_serialized_artifact_paths_must_remain_canonical_posix(self):
        payload = self._export()
        experiment, verifier = self._experiment_and_verifier(payload)
        tampered = [r"nested\metrics.json"]
        experiment["expected_artifacts"] = tampered
        verifier["verification"]["required_artifacts_by_task"][experiment["task_id"]] = tampered

        with self.assertRaisesRegex(ValueError, "canonical POSIX paths"):
            verify_export_completion(
                payload,
                task_statuses={experiment["task_id"]: "SUCCEEDED"},
                artifact_root="/tmp/evidence",
            )

    def test_status_contract_format_drift_is_rejected(self):
        payload = self._export()
        payload["status_contract"]["format"] = "newline-delimited text"
        experiment, _ = self._experiment_and_verifier(payload)

        with self.assertRaisesRegex(ValueError, "status format does not match"):
            verify_export_completion(
                payload,
                task_statuses={experiment["task_id"]: "SUCCEEDED"},
                artifact_root="/tmp/evidence",
            )

    def test_verifier_expected_artifacts_must_remain_empty(self):
        payload = self._export()
        experiment, verifier = self._experiment_and_verifier(payload)
        verifier["expected_artifacts"] = ["unexpected.json"]

        with self.assertRaisesRegex(ValueError, "verification task expected_artifacts must be empty"):
            verify_export_completion(
                payload,
                task_statuses={experiment["task_id"]: "SUCCEEDED"},
                artifact_root="/tmp/evidence",
            )

    def test_unmodified_export_still_reaches_runtime_evidence_check(self):
        payload = copy.deepcopy(self._export())
        experiment, _ = self._experiment_and_verifier(payload)

        ok, errors = verify_export_completion(
            payload,
            task_statuses={experiment["task_id"]: "SUCCEEDED"},
            artifact_root="/tmp/evidence",
        )

        self.assertFalse(ok)
        self.assertEqual(errors, ["artifact root does not exist or cannot be resolved: /tmp/evidence"])


if __name__ == "__main__":
    unittest.main()
