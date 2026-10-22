"""Adversarial adapter tests use fabricated fixtures, never scientific outcomes."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "NEUROGENESIS/successors/memory_v4/delivery/receipt_contract.py"
spec = importlib.util.spec_from_file_location("receipt_contract", FILE)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def run(tmp_path):
    put(tmp_path / "manifest.json", {"dataset": "unit", "configs": {"candidate": {}}})
    put(tmp_path / "summary.json", {"results": {"candidate": {}}})
    attempt = tmp_path / "attempts/unit__candidate"
    put(attempt / "result.json", {})
    (attempt / "trace.jsonl.gz").write_bytes(b"test fixture, not a real trace")
    (attempt / "checkpoint.json.gz").write_bytes(b"test fixture, not a checkpoint")
    exact = {"status": "VALID", "failed": 0, "methods": 1, "verified_trace_rows": 10,
             "scientific_comparison": "COMPLETE", "exact_replay": True}
    put(tmp_path / "verification-exact.json", exact)
    put(tmp_path / "audit.json", {"status": "VALID", "failed": 0})
    return tmp_path


def fake_verifier(run, lock, replay):
    assert replay is True
    return contract.load(run / "verification-exact.json")


def test_legacy_missing_hash_is_rejected_instead_of_none_equality(run):
    with pytest.raises(ValueError, match="binding missing"):
        contract.check_receipts(run, fake_verifier(run, None, True))


def test_new_binding_preserves_old_receipt_and_requires_actual_map(run):
    before = (run / "verification-exact.json").read_bytes()
    check = contract.verify_transfer_bound(run, Path("unused"), fake_verifier)
    result = contract.check_receipts(run, check)
    assert result["verification"]["artifact_map_sha256"] == contract.digest_map(contract.raw_artifact_map(run))
    assert (run / "verification-exact.json").read_bytes() == before
    assert "artifact_map_sha256" not in result["historical_verification"]


def test_mutation_during_replay_is_rejected(run):
    def mutate(root, lock, replay):
        check = fake_verifier(root, lock, replay)
        (root / "summary.json").write_text('{"tampered":true}')
        return check
    with pytest.raises(ValueError, match="changed during"):
        contract.verify_transfer_bound(run, Path("unused"), mutate)


def test_mutation_after_replay_is_rejected(run):
    check = contract.verify_transfer_bound(run, Path("unused"), fake_verifier)
    (run / "summary.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="binding mismatch"): contract.check_receipts(run, check)


@pytest.mark.parametrize("field,value", [("status", "INVALID"), ("failed", 1), ("failed", None),
    ("exact_replay", False), ("scientific_comparison", "INVALID")])
def test_unsuccessful_fresh_verification_cannot_be_attested(run, field, value):
    def bad(root, lock, replay):
        result = fake_verifier(root, lock, replay); result[field] = value; return result
    with pytest.raises(ValueError): contract.verify_transfer_bound(run, Path("unused"), bad)


def test_missing_attempt_file_is_rejected(run):
    (run / "attempts/unit__candidate/result.json").unlink()
    with pytest.raises(ValueError): contract.raw_artifact_map(run)


def test_extra_attempt_is_rejected(run):
    (run / "attempts/extra").mkdir()
    with pytest.raises(ValueError, match="coverage"): contract.raw_artifact_map(run)


def test_symlink_is_rejected(run):
    target = run / "attempts/unit__candidate/trace.jsonl.gz"
    target.unlink(); target.symlink_to(run / "summary.json")
    with pytest.raises(ValueError, match="unsafe"): contract.raw_artifact_map(run)


def test_path_traversal_is_rejected(run):
    put(run / "manifest.json", {"dataset": "../outside", "configs": {"candidate": {}}})
    with pytest.raises(ValueError, match="unsafe"): contract.raw_artifact_map(run)


def test_duplicate_and_nonfinite_json_rejected(tmp_path):
    f = tmp_path / "bad.json"
    for text in ('{"status":1,"status":2}', '{"value":NaN}'):
        f.write_text(text)
        with pytest.raises(ValueError): contract.load(f)


def test_regular_confirmation_map_must_be_present_and_equal(run):
    exact = contract.load(run / "verification-exact.json"); exact["artifact_map_sha256"] = "a" * 64
    put(run / "verification-exact.json", exact)
    assert contract.check_receipts(run, exact)["verification"]["status"] == "VALID"
    changed = dict(exact, artifact_map_sha256="b" * 64)
    with pytest.raises(ValueError, match="mismatch"): contract.check_receipts(run, changed)


def test_real_auditor_empty_failure_list_schema(run):
    put(run / "audit.json", {"status": "VALID", "failed": []})
    check = contract.verify_transfer_bound(run, Path("unused"), fake_verifier)
    assert contract.check_receipts(run, check)["audit"]["failed"] == []


@pytest.mark.parametrize("failures", (None, False, ["failure"], 1))
def test_auditor_failures_cannot_be_silently_accepted(run, failures):
    put(run / "audit.json", {"status": "VALID", "failed": failures})
    check = contract.verify_transfer_bound(run, Path("unused"), fake_verifier)
    with pytest.raises(ValueError, match="audit"): contract.check_receipts(run, check)
