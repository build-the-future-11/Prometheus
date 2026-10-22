#!/usr/bin/env python3
"""Verify that a saved analysis directory is regenerated exactly from its bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_analyzer(root: Path):
    spec = importlib.util.spec_from_file_location("analyze_bundle", root / "scripts" / "analyze_bundle.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load analysis generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(root: Path, bundle: Path, analysis_dir: Path) -> dict:
    analyzer = _load_analyzer(root)
    errors: list[str] = []
    if not (analysis_dir / "analysis.json").is_file():
        return {"schema_version": "neurogenesis.analysis-verification.v1", "status": "INVALID", "errors": ["analysis.json missing"]}
    with tempfile.TemporaryDirectory() as temporary:
        regenerated = Path(temporary) / "analysis"
        analyzer.write_outputs(analyzer.build_analysis(bundle), regenerated)
        expected = {path.relative_to(regenerated).as_posix(): _hash(path) for path in regenerated.rglob("*") if path.is_file()}
    observed = {path.relative_to(analysis_dir).as_posix(): _hash(path) for path in analysis_dir.rglob("*") if path.is_file()}
    if expected != observed:
        for path in sorted(set(expected) | set(observed)):
            if expected.get(path) != observed.get(path):
                errors.append(f"derived artifact mismatch: {path}")
    analysis = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    if analysis.get("claim_boundary") != "Descriptive analysis of a development-only conventional-baseline bundle; no candidate mechanism or protected evaluation was run.":
        errors.append("claim boundary drift")
    return {
        "schema_version": "neurogenesis.analysis-verification.v1",
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "derived_artifact_count": len(observed),
        "derived_artifact_map_sha256": hashlib.sha256(json.dumps(dict(sorted(observed.items())), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "input_artifact_map_sha256": analysis.get("input_artifact_map_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verification receipt: {args.output}")
    result = verify(args.root.resolve(), args.bundle, args.analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
