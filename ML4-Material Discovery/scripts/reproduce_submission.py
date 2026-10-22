"""Fail-closed release gate for exporting a numerical paper from a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED_OUTPUTS = {
    "results.json",
    "receipt.json",
    "artifact_manifest.json",
    "predictions/oof_predictions.csv",
    "tables/official_fold_metrics.csv",
    "tables/calibration.csv",
    "tables/abstention.csv",
    "tables/ranking.csv",
    "tables/generalization.csv",
    "figures/parity.png",
    "figures/calibration.png",
    "figures/abstention.png",
    "figures/ranking.png",
    "figures/subgroups.png",
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_release_candidate(root: Path, run_id: str) -> list[str]:
    run_dir = root / "runs" / run_id
    errors: list[str] = []
    if not run_dir.is_dir():
        return [f"missing run directory: {run_dir}"]
    receipt_path = run_dir / "receipt.json"
    if not receipt_path.is_file():
        return ["missing receipt.json"]
    receipt = _load(receipt_path)
    if receipt.get("status") != "COMPLETED":
        errors.append(f"run status is {receipt.get('status')!r}, not 'COMPLETED'")
    missing = sorted(relative for relative in REQUIRED_OUTPUTS if not (run_dir / relative).is_file())
    errors.extend(f"missing required output: {relative}" for relative in missing)
    if errors:
        return errors
    sys.path.insert(0, str(root / "src"))
    from ml4materials.experiment import verify_artifact_manifest

    manifest = verify_artifact_manifest(run_dir)
    if not manifest["ok"]:
        errors.extend(manifest["problems"])
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = verify_release_candidate(args.root.resolve(), args.run_id)
    if errors:
        print(json.dumps({"release_ready": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"release_ready": True, "run_id": args.run_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
