"""Protocol identity regressions; no model or scientific outcomes are executed."""
import hashlib
import json
import unittest
from unittest.mock import patch

from atlas.planner import build_manifest


QUESTION = "Does latent prediction improve robustness under distribution shift?"
SCHEME = "atlas.protocol-sha256.v1"


class PlannerIdentityTests(unittest.TestCase):
    def test_every_protocol_input_changes_identity(self):
        original = build_manifest(QUESTION)
        variants = [
            {"treatment": "different method"},
            {"control": "different baseline"},
            {"metric": "different_score"},
            {"minimum_improvement": 0.03},
            {"seeds": (1, 2)},
            {"perturbations": ("clean", "occlusion")},
        ]
        for kwargs in variants:
            with self.subTest(kwargs=kwargs):
                self.assertNotEqual(original.run_id, build_manifest(QUESTION, **kwargs).run_id)

    def test_question_changes_identity(self):
        self.assertNotEqual(build_manifest(QUESTION).run_id,
                            build_manifest(QUESTION + " on a new task?").run_id)

    def test_equivalent_whitespace_is_normalized_before_identity(self):
        self.assertEqual(build_manifest(QUESTION).run_id,
                         build_manifest("  " + QUESTION.replace(" ", "\n  ") + "  ").run_id)

    def test_creation_time_is_excluded(self):
        with patch("atlas.planner.datetime") as clock:
            clock.now.return_value.isoformat.return_value = "2026-09-07T00:00:00+00:00"
            first = build_manifest(QUESTION)
            clock.now.return_value.isoformat.return_value = "2026-09-08T00:00:00+00:00"
            second = build_manifest(QUESTION)
        self.assertNotEqual(first.metadata["created_at"], second.metadata["created_at"])
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.metadata["protocol_sha256"], second.metadata["protocol_sha256"])

    def test_exact_threshold_not_rounded_prose_controls_identity(self):
        first = build_manifest(QUESTION, minimum_improvement=0.020001)
        second = build_manifest(QUESTION, minimum_improvement=0.020002)
        self.assertEqual(first.hypothesis, second.hypothesis)
        self.assertNotEqual(first.run_id, second.run_id)

    def test_execution_matrix_order_is_retained(self):
        for first, second in [
            ({"seeds": (1, 2)}, {"seeds": (2, 1)}),
            ({"perturbations": ("clean", "shift")},
             {"perturbations": ("shift", "clean")}),
        ]:
            with self.subTest(first=first):
                self.assertNotEqual(build_manifest(QUESTION, **first).run_id,
                                    build_manifest(QUESTION, **second).run_id)

    def test_digest_can_be_recomputed_from_saved_protocol(self):
        manifest = build_manifest(QUESTION)
        payload = manifest.to_dict()
        payload.pop("run_id")
        payload.pop("metadata")
        encoded = json.dumps({"identity_scheme": SCHEME, "protocol": payload},
                             sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        expected = hashlib.sha256(encoded).hexdigest()
        self.assertEqual(manifest.metadata["run_id_scheme"], SCHEME)
        self.assertEqual(manifest.metadata["protocol_sha256"], expected)
        self.assertTrue(manifest.run_id.endswith("-" + expected[:16]))
        self.assertRegex(manifest.run_id, r"^[a-z0-9-]+-[0-9a-f]{16}$")

    def test_generated_protocol_semantics_remain_unchanged(self):
        manifest = build_manifest(QUESTION)
        self.assertEqual(manifest.primary_criterion.threshold, 0.02)
        self.assertEqual(manifest.primary_criterion.operator, ">=")
        self.assertEqual([e.seed for e in manifest.experiments[:5]], [17, 29, 43, 71, 101])
        self.assertEqual(len(manifest.experiments), 15)
        self.assertEqual(manifest.metadata["schema_version"], "atlas.run.v0.1")
        self.assertIn("Do not stop early", manifest.stopping_rule)
        manifest.validate()

    def test_nonfinite_thresholds_remain_invalid(self):
        for threshold in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                build_manifest(QUESTION, minimum_improvement=threshold)

    def test_new_manifest_does_not_mutate_an_existing_snapshot(self):
        first = build_manifest(QUESTION)
        snapshot = first.to_dict()
        build_manifest(QUESTION, seeds=(2, 3))
        self.assertEqual(snapshot, first.to_dict())


if __name__ == "__main__":
    unittest.main()
