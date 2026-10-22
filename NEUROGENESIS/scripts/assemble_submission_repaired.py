#!/usr/bin/env python3
"""Versioned layout repair around the preserved v1 submission assembler.

Creates a NEW package, keeps the original evidence bytes, restores the path used
by REPRODUCE.md, and hashes nested manifests rather than silently omitting them.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

REFERENCE_ASSEMBLER_BLOB = "8403d4b0491a70ee0f0d35689877c2606ae866d5"
BUNDLE = Path("results/development/shared-baselines-v1-20261019")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_new_package(output: Path) -> dict:
    output = output.resolve()
    old = output / "reproducibility/results"
    stage = output / "reproducibility/.results-layout-repair"
    if not old.is_dir() or old.is_symlink() or (old / "development").exists() or stage.exists():
        raise ValueError("expected a new, unrepaired v1 package")
    manifest_path = output / "manifest.json"
    original_bytes = manifest_path.read_bytes()
    manifest = json.loads(original_bytes)
    if manifest.get("schema_version") != "neurogenesis.submission-package.v1":
        raise ValueError("unexpected input package schema")
    # Verify original listed payload before moving any file.
    for relative, expected in manifest["files"].items():
        path = output / relative
        if path.is_symlink() or output not in path.resolve().parents or sha(path) != expected:
            raise ValueError(f"input package identity mismatch: {relative}")
    old.rename(stage)
    target = output / "reproducibility" / BUNDLE
    target.parent.mkdir(parents=True)
    stage.rename(target)
    manifest.update(schema_version="neurogenesis.submission-package.v1.1",
                    original_assembly_manifest_sha256=hashlib.sha256(original_bytes).hexdigest(),
                    repair_scope="layout and package manifest coverage only; original scientific evidence unchanged")
    manifest["files"] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink(): raise ValueError("symlink in assembled package")
        if path.is_file() and path != manifest_path:
            manifest["files"][path.relative_to(output).as_posix()] = sha(path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def assemble(root: Path, analysis: Path, output: Path):
    source = Path(__file__).with_name("assemble_submission.py")
    data = source.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != REFERENCE_ASSEMBLER_BLOB:
        raise ValueError("original assembler changed; revalidate this repair before proceeding")
    spec = importlib.util.spec_from_file_location("preserved_v1_assembler", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.assemble(root, analysis, output)
    return repair_new_package(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = assemble(a.root.resolve(), a.analysis.resolve(), a.output.resolve())
    print(json.dumps({"status": "CREATED", "schema": result["schema_version"], "files": len(result["files"])}))

if __name__ == "__main__": main()
