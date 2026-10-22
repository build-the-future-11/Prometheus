"""Command-line entry points for validation, development execution, and aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import write_aggregate
from .protocol import Protocol
from .runner import load_development_config, run_development_matrix


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurogenesis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate frozen protocol and config")
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--config", type=Path, required=True)

    run = subparsers.add_parser("run-development", help="run admitted baselines serially")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate", help="aggregate all retained attempts")
    aggregate.add_argument("--bundle", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        protocol = Protocol.load(args.protocol)
        config = load_development_config(args.config)
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "scope": protocol.status.value,
                    "protocol_sha256": protocol.digest,
                    "matrix_id": config["matrix_id"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-development":
        root = Path(__file__).resolve().parents[1]
        receipt = run_development_matrix(
            root=root,
            protocol_path=args.protocol,
            config_path=args.config,
            output_root=args.output,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["failed"] == 0 else 1
    if args.command == "aggregate":
        aggregate = write_aggregate(args.bundle, args.output)
        print(json.dumps(aggregate, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")

