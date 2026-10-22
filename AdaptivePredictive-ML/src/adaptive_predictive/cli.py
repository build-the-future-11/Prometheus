from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_series, synthetic_series, verify_run, write_run
from .model import DDFSConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-predictive")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("synthetic", help="run the non-evidentiary synthetic smoke study")
    synthetic.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify a retained run directory")
    verify.add_argument("--run", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "synthetic":
        config = DDFSConfig()
        spec = synthetic_series()
        result = run_series(spec, config, evaluate_from=600)
        write_run(args.output, [result], config, stage="synthetic-smoke")
        print(json.dumps(result["metrics"], indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        issues = verify_run(args.run)
        print(json.dumps({"run": str(args.run), "valid": not issues, "issues": issues}, sort_keys=True))
        return int(bool(issues))
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

