"""Command-line entry points for preparation, training/evaluation, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import load_data_contract, prepare_manifests
from .experiment import (
    create_artifact_manifest,
    run_study,
    verify_artifact_manifest,
)
from .inference import inspect_checkpoint, predict_formulas, rank_csv
from .successor import run_quantile_study
from .robust import run_robust, predict_robust
from .studies import (
    build_run_registry,
    compare_completed_runs,
    plan_study,
    preflight_study,
    write_preflight_receipt,
    write_run_registry,
)


def _root(value: str) -> Path:
    return Path(value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml4materials")
    parser.add_argument("--root", default=".", help="project root (default: current directory)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="validate pinned data and write manifests")
    run = subparsers.add_parser("run", help="train and execute the frozen development study")
    run.add_argument("--run-id", required=True)
    quantile = subparsers.add_parser("run-quantile", help="run the separate direct-quantile successor study")
    quantile.add_argument("--run-id", required=True)
    quantile.add_argument("--config", default="configs/studies/quantile-composition-v2.json")
    robust = subparsers.add_parser("run-robust", help="run frozen grouped robust-aggregation v3 development study")
    robust.add_argument("--run-id", required=True)
    robust.add_argument("--config", default="configs/studies/robust-aggregation-v3.json")
    robust_predict = subparsers.add_parser("predict-robust", help="predict from a v3 fold checkpoint")
    robust_predict.add_argument("--checkpoint", required=True)
    robust_predict.add_argument("--formula", action="append", required=True)

    predict = subparsers.add_parser("predict", help="predict formulas with calibrated uncertainty")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--formula", action="append", required=True)
    predict.add_argument("--target-lower", type=float, default=1.1)
    predict.add_argument("--target-upper", type=float, default=1.6)

    rank = subparsers.add_parser("rank-csv", help="rank a CSV candidate pool without labels")
    rank.add_argument("--checkpoint", required=True)
    rank.add_argument("--input", required=True)
    rank.add_argument("--output", required=True)
    rank.add_argument("--target-lower", type=float, default=1.1)
    rank.add_argument("--target-upper", type=float, default=1.6)

    inspect = subparsers.add_parser("inspect-checkpoint", help="summarize fitted feature importances")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--limit", type=int, default=20)

    verify = subparsers.add_parser("verify", help="verify a retained run artifact manifest")
    verify.add_argument("--run-id", required=True)
    manifest = subparsers.add_parser("manifest", help="regenerate a run artifact manifest")
    manifest.add_argument("--run-id", required=True)
    plan = subparsers.add_parser("plan", help="show a hash-bound study plan without allocating a run")
    plan.add_argument("--config", default="configs/study.json")
    preflight = subparsers.add_parser("preflight", help="validate inputs and resource admission before a run")
    preflight.add_argument("--config", default="configs/study.json")
    preflight.add_argument("--output")
    registry = subparsers.add_parser("registry", help="index retained runs without modifying them")
    registry.add_argument("--output")
    compare = subparsers.add_parser("compare", help="compare completed runs with identical inputs")
    compare.add_argument("--run-id", action="append", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root(args.root)
    if args.command == "prepare":
        payload = prepare_manifests(root, load_data_contract(root))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        print(run_study(root, args.run_id))
        return 0
    if args.command == "run-quantile":
        print(run_quantile_study(root, args.run_id, root / args.config))
        return 0
    if args.command == "run-robust":
        print(run_robust(root, args.run_id, root / args.config))
        return 0
    if args.command == "predict-robust":
        print(json.dumps(predict_robust(Path(args.checkpoint), args.formula), indent=2))
        return 0
    if args.command == "predict":
        rows = predict_formulas(
            Path(args.checkpoint),
            args.formula,
            (args.target_lower, args.target_upper),
        )
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if args.command == "rank-csv":
        rank_csv(
            Path(args.checkpoint),
            Path(args.input),
            Path(args.output),
            (args.target_lower, args.target_upper),
        )
        print(args.output)
        return 0
    if args.command == "inspect-checkpoint":
        print(json.dumps(inspect_checkpoint(Path(args.checkpoint), args.limit), indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        result = verify_artifact_manifest(root / "runs" / args.run_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.command == "manifest":
        payload = create_artifact_manifest(root / "runs" / args.run_id)
        print(json.dumps({"file_count": len(payload["files"])}, indent=2))
        return 0
    if args.command == "plan":
        print(json.dumps(plan_study(root, root / args.config), indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        if args.output:
            payload = write_preflight_receipt(root, root / args.config, Path(args.output))
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["status"] == "ADMITTED_AT_PREFLIGHT" else 1
        print(json.dumps(preflight_study(root, root / args.config), indent=2, sort_keys=True))
        return 0
    if args.command == "registry":
        if args.output:
            payload = write_run_registry(root, Path(args.output))
        else:
            payload = build_run_registry(root)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        print(json.dumps(compare_completed_runs(root, args.run_id), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())

