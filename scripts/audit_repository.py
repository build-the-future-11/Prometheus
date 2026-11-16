#!/usr/bin/env python3
"""Fail-closed tracked-tree inventory; works in full and sparse checkouts.

Inspects tracked source availability and CI contracts, never protected outcomes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ("AdaptivePredictive-ML", "CAUSAL-ALPHA", "Derived-Learning-ML", "FIN-ML",
            "META-RenormalizedEngramNeuralNetworks", "ML4-Biocomputation",
            "ML4-Material Discovery", "NEUROGENESIS", "PI-ML", "Project-32000", "SPARKING")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def parse_tree(text):
    entries = {}
    for item in text.split("\0"):
        if not item: continue
        meta, path = item.split("\t", 1)
        mode, kind, sha = meta.split()
        entries[path] = {"mode": mode, "kind": kind, "sha": sha}
    return entries


def inspect(root, files, workflows):
    findings = []; projects = []
    for name in PROJECTS:
        entry = root.get(name)
        if entry is None:
            findings.append({"severity": "error", "project": name, "code": "MISSING_PROJECT"})
            projects.append({"name": name, "state": "missing"}); continue
        if entry["mode"] == "160000":
            findings.append({"severity": "error", "project": name, "code": "GITLINK_SOURCE_UNAVAILABLE", "commit": entry["sha"]})
            if ".gitmodules" not in root:
                findings.append({"severity": "error", "project": name, "code": "MISSING_GITMODULES"})
            projects.append({"name": name, "state": "gitlink", "commit": entry["sha"]}); continue
        source = [p for p in files if p.startswith(name + "/") and p.endswith(".py") and "__pycache__" not in Path(p).parts]
        if not source: findings.append({"severity": "error", "project": name, "code": "NO_READABLE_PYTHON_SOURCE"})
        projects.append({"name": name, "state": "tree", "tracked_python_files": len(source)})
    for path, text in workflows.items():
        if any(token in text for token in ("atlas init-run", "atlas benchmark", "atlas-research-engine")):
            findings.append({"severity": "error", "project": ".github", "code": "LEGACY_ATLAS_WORKFLOW", "path": path})
    return {"schema_version": 2, "scope": "tracked source availability; not scientific result validation",
            "projects": projects, "findings": findings, "passed": not findings}


def audit():
    root = parse_tree(git("ls-tree", "-z", "HEAD"))
    files = parse_tree(git("ls-tree", "-r", "-z", "HEAD"))
    workflows = {p: git("show", "HEAD:" + p) for p in files if p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml"))}
    return dict(inspect(root, files, workflows), commit=git("rev-parse", "HEAD").strip())


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path); args = p.parse_args()
    result = audit(); text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text)
    else: sys.stdout.write(text)
    return 0 if result["passed"] else 1

if __name__ == "__main__": raise SystemExit(main())
