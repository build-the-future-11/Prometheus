from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .benchmarks import (
    build_benchmark_manifest,
    load_cases,
    load_predictions,
    score_exact_match,
    validate_benchmark,
)
from .percy import build_percy_export, load_json_object, verify_export_completion, write_percy_export
from .planner import build_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas", description="Atlas Research Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_run = subparsers.add_parser("init-run", help="Create a validated research run manifest")
    init_run.add_argument("--question", required=True, help="Research question")
    init_run.add_argument("--treatment", default="proposed method")
    init_run.add_argument("--control", default="strong baseline")
    init_run.add_argument("--metric", default="primary_score")
    init_run.add_argument("--minimum-improvement", type=float, default=0.02)
    init_run.add_argument("--seeds", default="17,29,43,71,101", help="Comma-separated non-negative integer seeds")
    init_run.add_argument(
        "--perturbations",
        default="in_distribution,distribution_shift,stress_test",
        help="Comma-separated evaluation conditions",
    )
    init_run.add_argument("--output", required=True, help="Destination JSON path")

    benchmark = subparsers.add_parser("benchmark", help="Validate and score deterministic benchmark packages")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)

    validate = benchmark_commands.add_parser("validate", help="Validate benchmark JSONL and optional manifest")
    validate.add_argument("--cases", required=True, help="Benchmark cases JSONL")
    validate.add_argument("--manifest", help="Optional benchmark manifest JSON")

    manifest = benchmark_commands.add_parser("manifest", help="Create a benchmark manifest bound to exact case bytes")
    manifest.add_argument("--cases", required=True, help="Benchmark cases JSONL")
    manifest.add_argument("--name", required=True)
    manifest.add_argument("--version", required=True)
    manifest.add_argument("--license", required=True, dest="license_name")
    manifest.add_argument("--task-description", required=True)
    manifest.add_argument("--output", required=True, help="Destination manifest JSON")

    score = benchmark_commands.add_parser("score", help="Score prediction JSONL with exact match")
    score.add_argument("--cases", required=True, help="Benchmark cases JSONL")
    score.add_argument("--predictions", required=True, help="Prediction JSONL")
    score.add_argument("--output", required=True, help="Destination score report JSON")

    export_percy = subparsers.add_parser("export-percy", help="Export an Atlas run as Percy-ready durable tasks")
    export_percy.add_argument("--manifest", required=True, help="Atlas run manifest JSON")
    export_percy.add_argument("--output", required=True, help="Destination Percy task JSON")
    export_percy.add_argument("--working-directory", required=True, help="Executor working directory")
    export_percy.add_argument(
        "--command-template",
        required=True,
        help=(
            "Experiment command template. Available placeholders: {run_id}, {experiment_id}, {experiment_name}, "
            "{seed}, {perturbation}, {treatment}, {control}, {artifact_dir}"
        ),
    )
    export_percy.add_argument("--artifact-root", help="Artifact root; defaults beside the export file")
    export_percy.add_argument("--status-file", help="Percy status JSON path; defaults beside the export file")
    export_percy.add_argument("--timeout-seconds", type=int, default=3600)
    export_percy.add_argument("--max-retries", type=int, default=1)
    export_percy.add_argument("--retry-backoff-seconds", type=int, default=30)
    export_percy.add_argument("--resource-class", default="heavy")
    export_percy.add_argument("--resource-slots", type=int, default=1)

    verify_percy = subparsers.add_parser("verify-percy", help="Fail closed on failed tasks or missing artifacts")
    verify_percy.add_argument("--export", required=True, help="Percy task export JSON")
    verify_percy.add_argument("--statuses", required=True, help="JSON object mapping experiment task IDs to statuses")
    verify_percy.add_argument("--artifact-root", required=True, help="Root containing one directory per Atlas experiment")
    return parser


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--seeds must contain comma-separated integers") from exc
    if not values:
        raise ValueError("--seeds must contain at least one seed")
    return values


def _parse_strings(raw: str, argument: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{argument} must contain at least one value")
    return values


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolved_contract_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    return Path(value).expanduser().resolve(strict=False)


def _validate_percy_verification_paths(
    export_payload: dict[str, object],
    *,
    statuses: str | Path,
    artifact_root: str | Path,
) -> None:
    status_contract = export_payload.get("status_contract")
    if not isinstance(status_contract, dict):
        raise ValueError("export payload status_contract is malformed")
    expected_statuses = _resolved_contract_path(status_contract.get("path"), "export.status_contract.path")
    actual_statuses = Path(statuses).expanduser().resolve(strict=False)
    if actual_statuses != expected_statuses:
        raise ValueError("verification statuses path does not match export status contract")

    tasks = export_payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("export payload tasks must be a non-empty list")
    verifiers = [task for task in tasks if isinstance(task, dict) and task.get("kind") == "verification"]
    if len(verifiers) != 1:
        raise ValueError("export payload must contain exactly one verification task")

    expected_root = _resolved_contract_path(
        verifiers[0].get("artifact_directory"),
        "verification.artifact_directory",
    )
    actual_root = Path(artifact_root).expanduser().resolve(strict=False)
    if actual_root != expected_root:
        raise ValueError("verification artifact root does not match export artifact contract")

    for task in tasks:
        if not isinstance(task, dict) or task.get("kind") != "experiment":
            continue
        task_id = str(task.get("task_id") or "<unknown>")
        metadata = task.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{task_id}: metadata is malformed")
        experiment_id = metadata.get("atlas_experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError(f"{task_id}: metadata.atlas_experiment_id must be a non-empty string")
        declared_directory = _resolved_contract_path(
            task.get("artifact_directory"),
            f"{task_id}.artifact_directory",
        )
        expected_directory = (expected_root / experiment_id).resolve(strict=False)
        if declared_directory != expected_directory:
            raise ValueError(f"{task_id}: artifact_directory does not match export artifact root")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-run":
        try:
            manifest = build_manifest(
                args.question,
                treatment=args.treatment,
                control=args.control,
                metric=args.metric,
                minimum_improvement=args.minimum_improvement,
                seeds=_parse_seeds(args.seeds),
                perturbations=_parse_strings(args.perturbations, "--perturbations"),
            )
        except ValueError as exc:
            parser.error(str(exc))

        _write_json(args.output, manifest.to_dict())
        print(f"wrote {manifest.run_id} with {len(manifest.experiments)} experiments to {args.output}")
        return 0

    if args.command == "benchmark":
        try:
            if args.benchmark_command == "validate":
                report = validate_benchmark(args.cases, args.manifest)
                print(json.dumps(report, indent=2, sort_keys=True))
                return 0

            if args.benchmark_command == "manifest":
                manifest = build_benchmark_manifest(
                    args.cases,
                    name=args.name,
                    version=args.version,
                    license_name=args.license_name,
                    task_description=args.task_description,
                )
                _write_json(args.output, manifest.to_dict())
                print(f"wrote benchmark manifest for {manifest.case_count} cases to {args.output}")
                return 0

            if args.benchmark_command == "score":
                report = score_exact_match(load_cases(args.cases), load_predictions(args.predictions))
                _write_json(args.output, report.to_dict())
                print(
                    f"scored {report.total} cases: {report.correct} correct, "
                    f"{report.missing} missing, accuracy={report.accuracy:.6f}"
                )
                return 0
        except ValueError as exc:
            parser.error(str(exc))

    if args.command == "export-percy":
        try:
            manifest = load_json_object(args.manifest)
            run_id = str(manifest.get("run_id") or "").strip()
            if not run_id:
                raise ValueError("manifest.run_id must be a non-empty string")

            output = Path(args.output).expanduser().resolve(strict=False)
            artifact_root = (
                Path(args.artifact_root).expanduser().resolve(strict=False)
                if args.artifact_root
                else output.parent / f"{run_id}-artifacts"
            )
            status_file = (
                Path(args.status_file).expanduser().resolve(strict=False)
                if args.status_file
                else output.parent / f"{run_id}-task-statuses.json"
            )
            verification_command = " ".join(
                [
                    "python -m atlas.cli verify-percy",
                    "--export",
                    shlex.quote(str(output)),
                    "--statuses",
                    shlex.quote(str(status_file)),
                    "--artifact-root",
                    shlex.quote(str(artifact_root)),
                ]
            )

            payload = build_percy_export(
                manifest,
                working_directory=args.working_directory,
                command_template=args.command_template,
                artifact_root=artifact_root,
                status_file=status_file,
                verification_command=verification_command,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                resources={"class": args.resource_class, "slots": args.resource_slots},
            )
            write_percy_export(payload, output)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(str(exc))

        print(
            f"wrote {payload['experiment_task_count']} experiment tasks + verification task "
            f"for {payload['atlas_run_id']} to {output}"
        )
        return 0

    if args.command == "verify-percy":
        try:
            export_payload = load_json_object(args.export)
            _validate_percy_verification_paths(
                export_payload,
                statuses=args.statuses,
                artifact_root=args.artifact_root,
            )
            statuses = load_json_object(args.statuses)
            ok, errors = verify_export_completion(
                export_payload,
                task_statuses={str(key): str(value) for key, value in statuses.items()},
                artifact_root=args.artifact_root,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"verification error: {exc}")
            return 2

        if ok:
            print("Percy verification passed: all experiment tasks succeeded and required artifacts are present.")
            return 0
        for error in errors:
            print(f"verification failed: {error}")
        return 1

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
