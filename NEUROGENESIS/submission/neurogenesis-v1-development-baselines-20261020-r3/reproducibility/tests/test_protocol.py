from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from neurogenesis.contracts import ContractError, ScopeStatus
from neurogenesis.interfaces import CandidateNotApproved, require_candidate_approval
from neurogenesis.protocol import Protocol, strict_json_load
from neurogenesis.runner import load_development_config


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_current_protocol_is_unresolved_and_development_only(self) -> None:
        protocol = Protocol.load(ROOT / "configs" / "protocol.json")
        self.assertIs(protocol.status, ScopeStatus.UNRESOLVED)
        self.assertIsNone(protocol.approved_mechanism)
        protocol.admit("development", "standard_sgd")
        with self.assertRaises(PermissionError):
            protocol.admit("evaluation", "standard_sgd")
        with self.assertRaises(PermissionError):
            protocol.admit("development", "learned_rule_candidate")

    def test_candidate_interfaces_fail_closed_until_one_scope_is_approved(self) -> None:
        for candidate_scope in (ScopeStatus.LEARNED_RULE, ScopeStatus.MEMORY_WEIGHT):
            with self.assertRaises(CandidateNotApproved):
                require_candidate_approval(ScopeStatus.UNRESOLVED, candidate_scope)
        with self.assertRaises(CandidateNotApproved):
            require_candidate_approval(ScopeStatus.LEARNED_RULE, ScopeStatus.MEMORY_WEIGHT)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"status": 1, "status": 2}', encoding="utf-8")
            with self.assertRaises(ContractError):
                strict_json_load(path)

    def test_registered_identity_sets_are_disjoint(self) -> None:
        config = load_development_config(ROOT / "configs" / "development.json")
        task_ids = [item["identity"]["task_id"] for item in config["tasks"]]
        meta_ids = [item["identity"]["task_id"] for item in config["meta_train_identities"]]
        self.assertTrue(set(task_ids).isdisjoint(meta_ids))
        self.assertEqual(len(task_ids), 5)


if __name__ == "__main__":
    unittest.main()

