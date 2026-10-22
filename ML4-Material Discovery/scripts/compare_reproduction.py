"""Compare a fresh full execution with the preserved original, without selection."""
from pathlib import Path
import argparse
import json
from scripts.verify_evidence import verify, read, digest, table, require
from scripts.verify_resampling import verify_resampling


TIMING_FIELDS = {"duration_seconds", "duration_shared_seconds", "fit_predict_seconds"}


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k not in TIMING_FIELDS}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def compare(root, original, reproduced):
    audits = [verify(root, run_id) for run_id in [original, reproduced]]
    resampling = [verify_resampling(root, run_id) for run_id in [original, reproduced]]
    a, b = [root / "runs" / run_id for run_id in [original, reproduced]]
    receipts = [read(run / "receipt.json") for run in [a, b]]
    require(receipts[0]["source_identity"] == receipts[1]["source_identity"], "executed source identities differ")
    require(receipts[0]["config_source_sha256"] == receipts[1]["config_source_sha256"], "configuration source identities differ")
    byte_files = ["config.json", "units.json", "splits.json", "predictions/oof.csv",
                  "predictions/ranking.csv", "predictions/holdouts.csv", "quantile_diagnostic.json",
                  "tables/metrics.csv", "tables/calibration.csv", "tables/ranking.csv", "tables/subgroups.csv"]
    identities = {name: {"original": digest(a / name), "reproduced": digest(b / name)} for name in byte_files}
    mismatches = [name for name, pair in identities.items() if pair["original"] != pair["reproduced"]]
    summary_equal = without_timing(read(a / "results.json")) == without_timing(read(b / "results.json"))
    table_equal = {name: without_timing(table(a / name)) == without_timing(table(b / name))
                   for name in ["tables/costs.csv", "tables/probes.csv", "tables/holdouts.csv"]}
    checkpoints = {p.name: digest(p) == digest(b / "models" / p.name) for p in sorted((a / "models").glob("*.joblib"))}
    exact = not mismatches and summary_equal and all(table_equal.values())
    return {"status": "EXACT_SCIENTIFIC_REPRODUCTION" if exact else "DIFFERENCES_REQUIRE_REVIEW",
            "original_run": original, "reproduction_run": reproduced,
            "executed_source_identity": receipts[0]["source_identity"]["sha256"],
            "byte_comparisons": identities, "byte_mismatches": mismatches,
            "scientific_summary_exact_excluding_timing": summary_equal,
            "tables_exact_excluding_timing": table_equal,
            "checkpoint_bytes_equal": checkpoints,
            "verification": audits, "resampling_verification": resampling,
            "runtime_seconds": [r["duration_seconds"] for r in receipts],
            "peak_rss_mib": [r["resources"]["peak_observed_rss_mib"] for r in receipts],
            "dependencies": [r["dependencies"] for r in receipts],
            "scope": "Fresh Kaggle execution by the same operator using identical source; not independent scientific replication or protected external confirmation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--original", default="kaggle-robust-aggregation-v3-r1")
    parser.add_argument("--reproduced", default="kaggle-robust-aggregation-v3-reproduction-r1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.root, args.original, args.reproduced)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"verification", "byte_comparisons"}}, indent=2))
    return 0 if result["status"] == "EXACT_SCIENTIFIC_REPRODUCTION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
