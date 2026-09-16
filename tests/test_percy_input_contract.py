import copy
import unittest

from atlas.percy import build_percy_export, verify_export_completion
from atlas.planner import build_manifest


class PercyInputContractTests(unittest.TestCase):
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
            "command_template": "python run.py --experiment {experiment_id} --output {artifact_dir}",
            "artifact_root": "/tmp/evidence",
            "status_file": "/tmp/task-statuses.json",
            "verification_command": "python -m atlas.cli verify-percy",
            "timeout_seconds": 120,
            "max_retries": 1,
            "retry_backoff_seconds": 5,
            "resources": {"class": "heavy", "slots": 1},
        }
        options.update(overrides)
        return build_percy_export(manifest or self._manifest(), **options)

    def test_manifest_text_fields_do_not_coerce_non_strings(self):
        mutations = (
            ("run_id", lambda manifest: manifest.__setitem__("run_id", 42)),
            ("experiment name", lambda manifest: manifest["experiments"][0].__setitem__("name", 42)),
            ("treatment", lambda manifest: manifest["experiments"][0].__setitem__("treatment", True)),
            (
                "required artifact",
                lambda manifest: manifest["experiments"][0].__setitem__("required_artifacts", (42,)),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                manifest = copy.deepcopy(self._manifest())
                mutate(manifest)
                with self.assertRaises(ValueError):
                    self._export(manifest)

    def test_boolean_seed_is_rejected_in_raw_manifest(self):
        manifest = copy.deepcopy(self._manifest())
        manifest["experiments"][0]["seed"] = True
        with self.assertRaisesRegex(ValueError, "seed must be a non-negative integer"):
            self._export(manifest)

    def test_execution_numeric_controls_require_actual_integers(self):
        cases = (
            ("timeout boolean", {"timeout_seconds": True}),
            ("timeout float", {"timeout_seconds": 1.5}),
            ("retry boolean", {"max_retries": False}),
            ("retry float", {"max_retries": 1.0}),
            ("backoff boolean", {"retry_backoff_seconds": True}),
            ("backoff float", {"retry_backoff_seconds": 0.5}),
        )
        for label, override in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self._export(**override)

    def test_execution_paths_must_be_absolute(self):
        cases = (
            ("working directory", {"working_directory": "workspace/project"}, "working_directory"),
            ("artifact root", {"artifact_root": "evidence"}, "artifact_root"),
            ("status file", {"status_file": "task-statuses.json"}, "status_file"),
        )
        for label, override, field in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, f"{field} must be an absolute path"):
                    self._export(**override)

    def test_execution_paths_are_canonicalized_before_serialization(self):
        payload = self._export(
            working_directory="/workspace/./project",
            artifact_root="/tmp/run/../evidence",
            status_file="/tmp/run/../task-statuses.json",
        )
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        verifier = next(task for task in payload["tasks"] if task["kind"] == "verification")
        self.assertEqual(experiment["working_directory"], "/workspace/project")
        self.assertEqual(verifier["working_directory"], "/workspace/project")
        self.assertEqual(verifier["artifact_directory"], "/tmp/evidence")
        self.assertEqual(payload["status_contract"]["path"], "/tmp/task-statuses.json")

    def test_resource_contract_rejects_coerced_slots_and_non_mapping_payloads(self):
        invalid_resources = (
            {"class": "heavy", "slots": "2"},
            {"class": "heavy", "slots": True},
            {"class": 7, "slots": 1},
            [("class", "heavy"), ("slots", 1)],
        )
        for resources in invalid_resources:
            with self.subTest(resources=resources):
                with self.assertRaises(ValueError):
                    self._export(resources=resources)

    def test_valid_integer_controls_are_preserved_as_integers(self):
        payload = self._export(
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0,
            resources={"class": "heavy", "slots": 2},
        )
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        self.assertEqual(experiment["timeout_seconds"], 1)
        self.assertIs(type(experiment["timeout_seconds"]), int)
        self.assertEqual(experiment["retry_policy"], {"max_retries": 0, "backoff_seconds": 0})
        self.assertIs(type(experiment["retry_policy"]["max_retries"]), int)
        self.assertIs(type(experiment["retry_policy"]["backoff_seconds"]), int)
        self.assertEqual(experiment["resources"], {"class": "heavy", "slots": 2})
        self.assertIs(type(experiment["resources"]["slots"]), int)

    def test_verifier_rejects_non_string_export_identity_instead_of_coercing(self):
        payload = self._export()
        payload["atlas_run_id"] = 123
        with self.assertRaisesRegex(ValueError, "export.atlas_run_id must be a non-empty string"):
            verify_export_completion(payload, task_statuses={}, artifact_root="/tmp/does-not-matter")


if __name__ == "__main__":
    unittest.main()
