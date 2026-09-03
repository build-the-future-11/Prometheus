from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
