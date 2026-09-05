from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

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

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {manifest.run_id} with {len(manifest.experiments)} experiments to {output}")
        return 0

    if args.command == "export-percy":
        try:
            manifest = load_json_object(args.manifest)
            run_id = str(manifest.get("run_id") or "").strip()
            if not run_id:
                raise ValueError("manifest.run_id must be a non-empty string")

            output = Path(args.output)
            artifact_root = Path(args.artifact_root) if args.artifact_root else output.parent / f"{run_id}-artifacts"
            status_file = Path(args.status_file) if args.status_file else output.parent / f"{run_id}-task-statuses.json"
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
