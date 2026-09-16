from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ".github/workflows/build-backend-floor.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/package-reproducibility.yml",
    ".github/workflows/workflow-supply-chain.yml",
)
REMOTE_ACTION = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")
USES_LINE = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
RUNNER_LINE = re.compile(r"^\s*runs-on:\s*([^\s#]+)", re.MULTILINE)


class WorkflowSupplyChainContractTests(unittest.TestCase):
    def _read(self, path: str) -> str:
        workflow = ROOT / path
        self.assertTrue(workflow.is_file(), f"missing protected workflow: {path}")
        return workflow.read_text(encoding="utf-8")

    def test_protected_workflows_pin_runner_and_remote_actions(self) -> None:
        for path in WORKFLOWS:
            with self.subTest(path=path):
                source = self._read(path)
                runners = RUNNER_LINE.findall(source)
                self.assertTrue(runners, f"{path} declares no runner")
                self.assertTrue(
                    all(runner == "ubuntu-24.04" for runner in runners),
                    f"{path} must pin every runner to ubuntu-24.04: {runners}",
                )
                self.assertNotIn("ubuntu-latest", source)

                actions = USES_LINE.findall(source)
                self.assertTrue(actions, f"{path} declares no actions")
                for action in actions:
                    if action.startswith("./"):
                        continue
                    self.assertRegex(
                        action,
                        REMOTE_ACTION,
                        f"{path} must pin remote action refs to immutable 40-hex SHAs: {action}",
                    )

    def test_protected_workflows_bind_checkout_to_exact_source(self) -> None:
        for path in WORKFLOWS:
            with self.subTest(path=path):
                source = self._read(path)
                self.assertIn("EXPECTED_SOURCE_SHA:", source)
                self.assertIn("github.event.pull_request.head.sha || github.sha", source)
                self.assertIn("ref: ${{ env.EXPECTED_SOURCE_SHA }}", source)
                self.assertIn("persist-credentials: false", source)
                self.assertIn("git rev-parse HEAD", source)

    def test_protected_workflows_remain_least_privilege_and_pr_safe(self) -> None:
        for path in WORKFLOWS:
            with self.subTest(path=path):
                source = self._read(path)
                self.assertRegex(source, r"(?m)^permissions:\s*\n\s+contents:\s*read\s*$")
                self.assertNotRegex(source, r"(?m)^\s*pull_request_target:\s*$")
                self.assertNotRegex(source, r"(?m)^\s*permissions:\s*write-all\s*$")


if __name__ == "__main__":
    unittest.main()
