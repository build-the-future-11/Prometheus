#!/usr/bin/env python3
"""Fail-closed, dependency-free repository research integrity audit.

Usage: python scripts/audit_repository.py --output audit.json
Runs only metadata checks; does not execute experiments or inspect protected outcomes.
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECTS = (
    "AdaptivePredictive-ML", "CAUSAL-ALPHA", "Derived-Learning-ML",
    "FIN-ML", "META-RenormalizedEngramNeuralNetworks",
    "ML4-Biocomputation", "ML4-Material Discovery", "NEUROGENESIS",
    "PI-ML", "Project-32000", "SPARKING",
)

def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)

def audit():
    entries = git("ls-tree", "HEAD").splitlines()
    root_entries = {}
    for line in entries:
        metadata, path = line.split("\t", 1)
        mode, kind, sha = metadata.split()
        root_entries[path] = {"mode": mode, "kind": kind, "sha": sha}
    modules = ROOT / ".gitmodules"
    findings = []
    projects = []
    for name in PROJECTS:
        entry = root_entries.get(name)
        if not entry:
            findings.append({"severity": "error", "project": name, "code": "MISSING_PROJECT"})
            projects.append({"name": name, "state": "missing"})
            continue
        if entry["mode"] == "160000":
            findings.append({"severity": "error", "project": name, "code": "GITLINK_SOURCE_UNAVAILABLE", "commit": entry["sha"]})
            if not modules.exists():
                findings.append({"severity": "error", "project": name, "code": "MISSING_GITMODULES"})
            projects.append({"name": name, "state": "gitlink", "commit": entry["sha"]})
            continue
        path = ROOT / name
        source = list(path.rglob("*.py")) if path.is_dir() else []
        readable = [p for p in source if "__pycache__" not in p.parts]
        if not readable:
            findings.append({"severity": "error", "project": name, "code": "NO_READABLE_PYTHON_SOURCE"})
        projects.append({"name": name, "state": "tree", "python_files": len(readable)})
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        data = path.read_text(encoding="utf-8")
        if "atlas init-run" in data or "atlas benchmark" in data or "atlas-research-engine" in data:
            findings.append({"severity": "error", "project": ".github", "code": "LEGACY_ATLAS_WORKFLOW", "path": str(path.relative_to(ROOT))})
    return {
        "schema_version": 1,
        "commit": git("rev-parse", "HEAD").strip(),
        "scope": "static repository packaging and source availability; no scientific result validation",
        "projects": projects,
        "findings": findings,
        "passed": not any(f["severity"] == "error" for f in findings),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
