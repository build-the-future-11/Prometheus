"""Exercise evidence corruptions without editing retained artifacts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from reconcile_transfer_evidence import reconcile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, run = args.source.resolve(), args.run.resolve()
    sys.path.insert(0, str(source))
    from transfer import verify as legacy_verify
    cases = []
    for mutation in ("untampered_control", "false_support", "omitted_control", "wrong_units", "admission_refused",
                     "stale_audit_digest", "stale_exact_receipt"):
        with tempfile.TemporaryDirectory(prefix="neurogenesis-adversarial-", dir=args.output.parent) as name:
            altered = Path(name)
            for path in run.iterdir():
                if path.is_file():
                    shutil.copy2(path, altered / path.name)
            (altered / "attempts").mkdir()
            for path in (run / "attempts").iterdir():
                if mutation == "omitted_control" and path.name.endswith("__uniform_bank"):
                    continue
                shutil.copytree(path, altered / "attempts" / path.name, copy_function=os.link)
            manifest = json.loads((altered / "manifest.json").read_text())
            summary = json.loads((altered / "summary.json").read_text())
            if mutation == "false_support":
                summary["descriptive_support"] = not summary["descriptive_support"]
            elif mutation == "omitted_control":
                del manifest["configs"]["uniform_bank"]
                del summary["results"]["uniform_bank"]
            elif mutation == "wrong_units":
                manifest["scale_squared"] = 1.0
            elif mutation == "admission_refused":
                admission = json.loads((altered / "admission.json").read_text())
                admission["allowed"] = False
                (altered / "admission.json").write_text(json.dumps(admission))
            elif mutation == "stale_audit_digest":
                audit = json.loads((altered / "audit.json").read_text())
                audit["artifact_map_sha256"] = "0" * 64
                (altered / "audit.json").write_text(json.dumps(audit))
            elif mutation == "stale_exact_receipt":
                exact = json.loads((altered / "verification-exact.json").read_text())
                exact["utc"] = "2000-01-01T00:00:00+00:00"
                (altered / "verification-exact.json").write_text(json.dumps(exact))
            if mutation in {"omitted_control", "wrong_units"}:
                (altered / "manifest.json").write_text(json.dumps(manifest))
            if mutation in {"false_support", "omitted_control"}:
                (altered / "summary.json").write_text(json.dumps(summary))
            legacy = legacy_verify(altered, source / "runs" / "freeze.json", replay=False)
            rejected = False
            reason = None
            try:
                reconcile(source, altered, args.execution_receipt)
            except ValueError as error:
                rejected, reason = True, str(error)
            cases.append({"mutation": mutation, "legacy_status": legacy["status"],
                          "legacy_methods": legacy["methods"],
                          "supplemental_rejected": rejected, "reason": reason})
            if rejected != (mutation != "untampered_control"):
                raise AssertionError(f"Unexpected supplemental result for {mutation}: rejected={rejected}, reason={reason}")
    report = {"status": "PASS", "cases": cases, "cases_passed": len(cases),
              "corruptions_rejected": len(cases) - 1, "untampered_control_accepted": True,
              "retained_evidence_modified": False, "model_outcomes_executed": False,
              "scope": "Corruption checks against copies of Air metadata with read-only references to raw artifacts",
              "utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
