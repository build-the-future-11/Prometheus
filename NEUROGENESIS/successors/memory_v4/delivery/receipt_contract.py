"""Bind legacy v4 transfer verification to raw artifacts without rewriting receipts.

The frozen transfer verifier has no artifact_map_sha256 field. This adapter adds
an explicitly versioned binding only after a fresh exact replay, and rejects a
run that changes during verification. It never repairs or fabricates outcomes.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

DIGEST = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def load(path: Path) -> dict:
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result: raise ValueError(f"duplicate JSON key: {k}")
            result[k] = v
        return result
    def invalid(value): raise ValueError(f"nonfinite JSON: {value}")
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(data, dict): raise ValueError("receipt must be an object")
    return data


def digest_map(mapping: dict) -> str:
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def raw_artifact_map(run: Path) -> dict[str, str]:
    run = Path(run).resolve()
    manifest = load(run / "manifest.json")
    dataset = manifest.get("dataset")
    configs = manifest.get("configs")
    if not isinstance(dataset, str) or not NAME.fullmatch(dataset): raise ValueError("unsafe dataset name")
    if not isinstance(configs, dict) or not configs: raise ValueError("missing configs")
    if any(not isinstance(n, str) or not NAME.fullmatch(n) for n in configs): raise ValueError("unsafe method name")
    attempts = run / "attempts"
    if attempts.is_symlink() or not attempts.is_dir(): raise ValueError("invalid attempts directory")
    expected = {f"{dataset}__{n}" for n in configs}
    observed = {p.name for p in attempts.iterdir()}
    if observed != expected: raise ValueError("attempt coverage mismatch")
    paths = [run / "manifest.json", run / "summary.json"]
    for name in sorted(expected):
        attempt = attempts / name
        if attempt.is_symlink() or not attempt.is_dir(): raise ValueError("invalid attempt directory")
        paths.extend(attempt / n for n in ("result.json", "trace.jsonl.gz", "checkpoint.json.gz"))
    output = {}
    for path in paths:
        if path.is_symlink() or not path.is_file() or run not in path.resolve().parents:
            raise ValueError(f"unsafe or missing artifact: {path.name}")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
        output[path.relative_to(run).as_posix()] = h.hexdigest()
    return output


def verify_transfer_bound(run: Path, lock: Path, verifier: Callable) -> dict:
    before = raw_artifact_map(run)
    check = verifier(run, lock, replay=True)
    after = raw_artifact_map(run)
    if before != after: raise ValueError("artifacts changed during exact verification")
    if (check.get("status") != "VALID" or check.get("scientific_comparison") != "COMPLETE"
            or type(check.get("failed")) is not int or check["failed"] != 0
            or check.get("exact_replay") is not True):
        raise ValueError("fresh exact transfer replay did not pass")
    return dict(check, artifact_map_sha256=digest_map(before),
                artifact_binding_schema="prometheus.transfer-binding.v1",
                artifact_map=before,
                binding_origin="fresh_exact_replay_not_a_rewritten_historical_receipt")


def check_receipts(run: Path, check: dict) -> dict:
    exact = load(run / "verification-exact.json")
    audit = load(run / "audit.json")
    for value in (check, exact):
        if value.get("status") != "VALID" or type(value.get("failed")) is not int or value["failed"] != 0:
            raise ValueError("verification did not explicitly pass")
    # The retained arithmetic auditor uses an empty failure list, not a count.
    failures = audit.get("failed")
    audit_pass = (type(failures) is int and failures == 0) or (type(failures) is list and not failures)
    if audit.get("status") != "VALID" or not audit_pass:
        raise ValueError("audit did not explicitly pass")
    if exact.get("exact_replay") is not True: raise ValueError("historical exact replay missing")
    digest = check.get("artifact_map_sha256")
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ValueError("fresh artifact binding missing or malformed")
    if check.get("artifact_binding_schema") == "prometheus.transfer-binding.v1":
        if check.get("exact_replay") is not True or check.get("scientific_comparison") != "COMPLETE":
            raise ValueError("legacy transfer needs fresh exact replay")
        for key in ("methods", "failed", "verified_trace_rows"):
            if type(check.get(key)) is not int or check[key] < 0 or exact.get(key) != check[key]:
                raise ValueError(f"historical/fresh transfer mismatch: {key}")
        if check["methods"] < 1 or check["verified_trace_rows"] < 1:
            raise ValueError("empty transfer evidence")
        actual = raw_artifact_map(run)
        if check.get("artifact_map") != actual or digest_map(actual) != digest:
            raise ValueError("raw artifact binding mismatch")
    else:
        old = exact.get("artifact_map_sha256")
        if not isinstance(old, str) or not DIGEST.fullmatch(old) or old != digest:
            raise ValueError("historical artifact binding mismatch")
    return {"verification": check, "historical_verification": exact, "audit": audit}
