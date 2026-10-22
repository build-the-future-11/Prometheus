"""Command-line entry points for generation, execution, and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import write_analysis
from .audit import write_claim_status, write_global_ridge_audit
from .experiment import execute_experiment, load_config
from .report import write_report
from .tasks import generate_from_config, write_tasks
from .verification import verify_manifest, verify_saved_evaluation, write_verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="derived-learning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="materialize deterministic task data")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="run selection and the locked evaluation matrix")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--project-root", type=Path, default=Path.cwd())

    report = subparsers.add_parser("report", help="generate Markdown from saved outputs")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="independently recompute saved terminal evaluation metrics")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify every hash bound by a run manifest")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    analyze = subparsers.add_parser("analyze", help="generate statistical summaries and SVG figures from a finalized run")
    analyze.add_argument("--run-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)

    audit = subparsers.add_parser("audit-baselines", help="compare the analytic rule with global ridge")
    audit.add_argument("--run-dir", type=Path, required=True)
    audit.add_argument("--project-root", type=Path, default=Path.cwd())
    audit.add_argument("--output", type=Path, required=True)

    claims = subparsers.add_parser("claim-status", help="generate an evidence-bound claim admission receipt")
    claims.add_argument("--run-dir", type=Path, required=True)
    claims.add_argument("--baseline-audit", type=Path)
    claims.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "generate":
        config = load_config(arguments.config)
        tasks = generate_from_config(config)
        write_tasks(tasks, arguments.output)
        print(json.dumps({"status": "ok", "task_count": len(tasks), "output": str(arguments.output)}))
        return 0
    if arguments.command == "run":
        manifest = execute_experiment(arguments.config, arguments.output_dir, arguments.project_root.resolve())
        print(json.dumps({"status": manifest["status"], "output_dir": str(arguments.output_dir)}))
        return 0
    if arguments.command == "report":
        write_report(arguments.run_dir, arguments.output)
        print(json.dumps({"status": "ok", "output": str(arguments.output)}))
        return 0
    if arguments.command == "evaluate":
        payload = verify_saved_evaluation(arguments.run_dir)
        write_verification(payload, arguments.output)
        print(json.dumps({"status": payload["status"], "checked_result_count": payload["checked_result_count"]}))
        return 0
    if arguments.command == "verify":
        payload = verify_manifest(arguments.run_dir)
        write_verification(payload, arguments.output)
        print(json.dumps({"status": payload["status"], "checked_output_count": len(payload["checked"])}))
        return 0
    if arguments.command == "analyze":
        payload = write_analysis(arguments.run_dir, arguments.output_dir)
        print(json.dumps({"status": "ok", "splits": sorted(payload["by_split"]), "output_dir": str(arguments.output_dir)}))
        return 0
    if arguments.command == "audit-baselines":
        payload = write_global_ridge_audit(arguments.run_dir, arguments.project_root, arguments.output)
        print(json.dumps({"status": payload["status"], "equivalence": payload["numerical_equivalence_status"], "output": str(arguments.output)}))
        return 0
    if arguments.command == "claim-status":
        payload = write_claim_status(arguments.run_dir, arguments.output, arguments.baseline_audit)
        print(json.dumps({"disposition": payload["disposition"], "submission_ready": payload["submission_ready"], "output": str(arguments.output)}))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

