"""Command-line interface for the complete SPARKING workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import load_data, prepare_data
from .evidence import ResourceGuard, verify_manifest, write_json
from .experiment import run_confirmatory, run_development


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparking")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="download and hash-validate the pinned dataset")
    preflight = subparsers.add_parser("preflight", help="validate inputs and resource admission")
    preflight.add_argument("--output", default="artifacts/preflight.json")
    develop = subparsers.add_parser("develop", help="run the frozen development matrix")
    develop.add_argument("--run-id", required=True)
    confirm = subparsers.add_parser("confirm", help="run the frozen protected evaluation")
    confirm.add_argument("--run-id", required=True)
    verify = subparsers.add_parser("verify", help="verify a retained run manifest")
    verify.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.command == "prepare":
        result = prepare_data(root)
    elif args.command == "preflight":
        protocol = json.loads((root / "protocols" / "development-v1.json").read_text())
        guard = ResourceGuard(protocol["max_rss_mib"], protocol["require_no_rising_pageout"])
        guard.check("admission")
        formulas, targets = load_data(root)
        guard.check("data_validation")
        result = {
            "schema_version": 1,
            "status": "ADMITTED",
            "rows": len(formulas),
            "finite_targets": bool(targets.size and all(map(lambda value: value == value, targets))),
            "resource_observations": guard.observations,
        }
        write_json(root / args.output, result)
    elif args.command == "develop":
        result = run_development(root, args.run_id)
    elif args.command == "confirm":
        result = run_confirmatory(root, args.run_id)
    elif args.command == "verify":
        result = verify_manifest(root / "runs" / args.run_id)
    else:
        raise AssertionError(args.command)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")

