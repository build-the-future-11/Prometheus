"""Synthetic filesystem regressions; no experiment commands are executed."""

import tempfile
import unittest
from pathlib import Path

from atlas.percy import build_percy_export, verify_export_completion


class PercyArtifactOwnershipTests(unittest.TestCase):
    def _fixture(self, root):
        manifest = {
            "run_id": "synthetic-artifact-ownership",
            "experiments": [
                {
                    "name": name,
                    "seed": 0,
                    "perturbation": "fixture",
                    "treatment": "fixture",
                    "control": "fixture",
                    "required_artifacts": ["nested/metrics.json"],
                }
                for name in ("alpha", "beta")
            ],
        }
        payload = build_percy_export(
            manifest,
            working_directory=str(root),
            command_template="python fixture_only.py",
            artifact_root=root,
            status_file=root / "statuses.json",
            verification_command="python verify_fixture_only.py",
        )
        statuses = {}
        for task in payload["tasks"]:
            if task["kind"] != "experiment":
                continue
            statuses[task["task_id"]] = "SUCCEEDED"
            target = root / task["metadata"]["atlas_experiment_id"] / "nested/metrics.json"
            target.parent.mkdir(parents=True)
            target.write_text('{"synthetic_fixture": true}\n', encoding="utf-8")
        return payload, statuses

    def _verify(self, payload, statuses, root):
        return verify_export_completion(payload, task_statuses=statuses, artifact_root=root)

    def _assert_ownership_rejected(self, payload, statuses, root):
        ok, errors = self._verify(payload, statuses, root)
        self.assertFalse(ok)
        self.assertTrue(any("artifact escapes experiment root" in error for error in errors), errors)

    def test_independent_regular_artifacts_succeed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            self.assertEqual(self._verify(payload, statuses, root), (True, []))

    def test_sibling_file_symlink_cannot_satisfy_another_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            target = root / "beta/nested/metrics.json"
            target.unlink()
            target.symlink_to(root / "alpha/nested/metrics.json")
            self._assert_ownership_rejected(payload, statuses, root)

    def test_sibling_experiment_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            (root / "beta/nested/metrics.json").unlink()
            (root / "beta/nested").rmdir()
            (root / "beta").rmdir()
            (root / "beta").symlink_to(root / "alpha", target_is_directory=True)
            self._assert_ownership_rejected(payload, statuses, root)

    def test_sibling_nested_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            (root / "beta/nested/metrics.json").unlink()
            (root / "beta/nested").rmdir()
            (root / "beta/nested").symlink_to(root / "alpha/nested", target_is_directory=True)
            self._assert_ownership_rejected(payload, statuses, root)

    def test_common_root_file_cannot_satisfy_an_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            shared = root / "shared.json"
            shared.write_text("fixture", encoding="utf-8")
            target = root / "beta/nested/metrics.json"
            target.unlink()
            target.symlink_to(shared)
            self._assert_ownership_rejected(payload, statuses, root)

    def test_symlink_within_own_experiment_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            target = root / "beta/nested/metrics.json"
            retained = root / "beta/retained.json"
            target.rename(retained)
            target.symlink_to("../retained.json")
            self.assertEqual(self._verify(payload, statuses, root), (True, []))

    def test_caller_supplied_root_symlink_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "real"
            root.mkdir()
            payload, statuses = self._fixture(root)
            alias = parent / "alias"
            alias.symlink_to(root, target_is_directory=True)
            self.assertEqual(self._verify(payload, statuses, alias), (True, []))

    def test_external_symlink_keeps_existing_root_escape_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            outside = Path(outside_tmp) / "metrics.json"
            outside.write_text("outside fixture", encoding="utf-8")
            target = root / "beta/nested/metrics.json"
            target.unlink()
            target.symlink_to(outside)
            ok, errors = self._verify(payload, statuses, root)
            self.assertFalse(ok)
            self.assertTrue(any("artifact escapes artifact root" in error for error in errors))

    def test_missing_artifact_remains_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            (root / "beta/nested/metrics.json").unlink()
            ok, errors = self._verify(payload, statuses, root)
            self.assertFalse(ok)
            self.assertTrue(any("missing required artifact" in error for error in errors))

    def test_failed_dependency_cannot_be_rescued_by_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            statuses[next(iter(statuses))] = "FAILED"
            ok, errors = self._verify(payload, statuses, root)
            self.assertFalse(ok)
            self.assertTrue(any("dependency status is FAILED" in error for error in errors))

    def test_task_identity_protection_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, statuses = self._fixture(root)
            experiment = payload["tasks"][0]
            verifier = payload["tasks"][-1]
            original = experiment["task_id"]
            forged = "atlas-experiment-0000000000000000"
            experiment["task_id"] = forged
            statuses[forged] = statuses.pop(original)
            verifier["dependencies"] = [forged if key == original else key for key in verifier["dependencies"]]
            mapping = verifier["verification"]["required_artifacts_by_task"]
            mapping[forged] = mapping.pop(original)
            with self.assertRaisesRegex(ValueError, "deterministic stable identity"):
                self._verify(payload, statuses, root)


if __name__ == "__main__":
    unittest.main()
