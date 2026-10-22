from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run, verify
from .report import generate


def main() -> None:
    parser = argparse.ArgumentParser(prog="piml")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="execute one immutable experiment")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("output", type=Path)
    verify_parser = commands.add_parser("verify", help="verify a retained run")
    verify_parser.add_argument("run_dir", type=Path)
    report_parser = commands.add_parser("report", help="generate paper-bound results")
    report_parser.add_argument("run_dir", type=Path)
    report_parser.add_argument("paper_dir", type=Path, nargs="?", default=Path("paper"))
    args = parser.parse_args()
    if args.command == "run":
        result = run(json.loads(args.config.read_text()), args.output)
    elif args.command == "verify":
        result = verify(args.run_dir)
    else:
        result = generate(args.run_dir, args.paper_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))

