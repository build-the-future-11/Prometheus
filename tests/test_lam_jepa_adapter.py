import tempfile
import unittest
from pathlib import Path

from atlas.lam_jepa_adapter import (
    RETAINED_VERDICT,
    _create_run_dir,
    assert_forbidden_absent,
    build_execution_plan,
    dry_run_plan,
    load_integration,
)


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "lam_jepa_arc_v3.json"


class LamJepaAdapterTests(unittest.TestCase):
    def test_checked_in_integration_validates_and_preserves_verdict(self):
        config = load_integration(INTEGRATION)
        self.assertEqual(config.retained_verdict, RETAINED_VERDICT)
        self.assertFalse(config.raw["scientific_boundary"]["allow_test_split"])
        self.assertEqual(set(config.raw["scientific_boundary"]["allowed_splits"]), {"train", "validation"})

    def test_dry_run_constructs_frozen_command_order_without_execution(self):
        config = load_integration(INTEGRATION)
        steps = build_execution_plan(config)
        self.assertEqual(
            [step["name"] for step in steps],
            ["install", "download", "protocol_preflight", "producer", "verifier"],
        )
        self.assertEqual(steps[2]["command"], config.raw["protocol_preflight"])
        self.assertEqual(steps[3]["command"], config.raw["producer"]["command"])
        self.assertEqual(steps[4]["command"], config.raw["verifier"]["command"])

        plan = dry_run_plan(INTEGRATION)
        self.assertEqual(plan["source_ref"], config.source_ref)
        self.assertEqual(plan["retained_verdict"], RETAINED_VERDICT)
        self.assertEqual(plan["producer_output"], "ci-evidence/arc-protocol-v3-full-controls-validation.json")
        self.assertEqual(
            plan["verifier_output"],
            "ci-evidence/arc-protocol-v3-full-controls-validation-verification.json",
        )

    def test_forbidden_test_artifact_guard_fails_closed(self):
        config = load_integration(INTEGRATION)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assert_forbidden_absent(root, config)
            forbidden = root / config.forbidden_artifact
            forbidden.parent.mkdir(parents=True, exist_ok=True)
            forbidden.write_bytes(b"forbidden")
            with self.assertRaisesRegex(RuntimeError, "forbidden ARC test artifact"):
                assert_forbidden_absent(root, config)

    def test_fresh_run_directory_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _create_run_dir(root, "a" * 40, run_id="fixed-run")
            self.assertTrue(first.is_dir())
            with self.assertRaises(FileExistsError):
                _create_run_dir(root, "a" * 40, run_id="fixed-run")


if __name__ == "__main__":
    unittest.main()
