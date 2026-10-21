#!/usr/bin/env python3
"""Independently recompute v2 metrics and validate the retained run bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sparking.evidence import verify_manifest, write_json
from sparking.metrics import regression_metrics


METRICS = ("mae_ev", "rmse_ev", "median_ae_ev", "bias_ev")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = verify_manifest(run_dir)
    results = json.loads((run_dir / "results.json").read_text())
    if results["status"] != "SUCCESS":
        raise ValueError("run did not finish successfully")
    if results["protected_confirmatory_outcomes_observed"]:
        raise ValueError("protected outcomes were observed in a development run")
    comparisons = 0
    for element in ("Cu", "Ag", "B"):
        with np.load(run_dir / f"{element}-predictions.npz") as arrays:
            truth = arrays["target_ev"]
            for row in (item for item in results["rows"] if item["element"] == element):
                key = f"prediction_{row['model']}"
                if key not in arrays:
                    raise ValueError(f"missing raw prediction array {key}")
                observed = regression_metrics(truth, arrays[key])
                for metric in METRICS:
                    if not np.isclose(float(row[metric]), float(observed[metric]), rtol=0.0, atol=1e-10):
                        raise ValueError(f"metric drift for {element}/{row['model']}/{metric}")
                    comparisons += 1
                if row["model"].startswith("risk_gate_t"):
                    weight_key = f"weight_{row['model']}"
                    weights = arrays[weight_key]
                    active = float(np.mean(weights < 1.0 - 1e-12))
                    if not np.isclose(active, float(row["gate_active_fraction"]), rtol=0.0, atol=1e-12):
                        raise ValueError(f"gate activity drift for {element}/{row['model']}")
    receipt = json.loads((run_dir / "receipt.json").read_text())
    gpu = json.loads((run_dir / "gpu-admission.json").read_text())
    verification = {
        "status": "VERIFIED",
        "manifest": manifest,
        "metric_comparisons": comparisons,
        "scientific_disposition": results["scientific_disposition"],
        "protected_outcomes_observed": False,
        "run_receipt_status": receipt["status"],
        "gpu_admitted": gpu["exit_code"] == 0 and bool(gpu["stdout"]),
    }
    destination = args.output or run_dir.parent / f"{run_dir.name}-verification.json"
    write_json(destination, verification)
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
