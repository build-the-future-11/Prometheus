import copy
import unittest

from atlas.percy import build_percy_export


class PercyArtifactPathCanonicalizationTests(unittest.TestCase):
    def _manifest(self):
        return {
            "run_id": "artifact-path-canonicalization",
            "experiments": [
                {
                    "name": "clean",
                    "seed": 0,
                    "perturbation": "fixture",
                    "treatment": "fixture",
                    "control": "fixture",
                    "required_artifacts": ["nested/metrics.json"],
                }
            ],
        }

    def _export(self, manifest):
        return build_percy_export(
            manifest,
            working_directory="/workspace/project",
            command_template="python fixture_only.py --output {artifact_dir}",
            artifact_root="/tmp/evidence",
            status_file="/tmp/statuses.json",
            verification_command="python -m atlas.cli verify-percy",
        )

    def test_backslash_artifact_path_is_serialized_canonically(self):
        manifest = self._manifest()
        manifest["experiments"][0]["required_artifacts"] = [r"nested\metrics.json"]

        payload = self._export(manifest)
        experiment = next(task for task in payload["tasks"] if task["kind"] == "experiment")
        verifier = next(task for task in payload["tasks"] if task["kind"] == "verification")

        self.assertEqual(experiment["expected_artifacts"], ["nested/metrics.json"])
        self.assertEqual(
            verifier["verification"]["required_artifacts_by_task"][experiment["task_id"]],
            ["nested/metrics.json"],
        )

    def test_separator_aliases_are_rejected_as_duplicate_artifacts(self):
        manifest = copy.deepcopy(self._manifest())
        manifest["experiments"][0]["required_artifacts"] = [
            "nested/metrics.json",
            r"nested\metrics.json",
        ]

        with self.assertRaisesRegex(ValueError, "duplicate required artifact: nested/metrics.json"):
            self._export(manifest)


if __name__ == "__main__":
    unittest.main()
