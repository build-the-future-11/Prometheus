"""Recompute frozen random rankings and bootstrap bounds from raw predictions.

Uses NumPy solely for the specified PCG64 stream and percentile convention;
does not import the model, training code, or production statistical functions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from scripts.verify_evidence import read, table, require, close, verify


def audit_resamples(rows, units, config, results):
    keys = [u["canonical"] for u in units]
    by_key = {(int(r["repeat"]), r["canonical"]): r for r in rows}
    paired = []
    for key in keys:
        losses = []
        for repeat in range(len(config["split_seeds"])):
            r = by_key[repeat, key]
            losses.append(abs(float(r["target"]) - float(r["et_median"])) -
                          abs(float(r["target"]) - float(r["et_mean"])))
        paired.append(sum(losses) / len(losses))
    paired = np.asarray(paired)
    generator = np.random.default_rng(20261021)
    samples = np.empty(config["bootstrap_repetitions"])
    for i in range(len(samples)):
        indices = generator.integers(0, len(keys), size=len(keys))
        samples[i] = np.average(paired[indices])
    interval = np.percentile(samples, [2.5, 97.5])
    for actual, expected in zip(results["primary"]["percentile_95_interval_ev"], interval):
        close(actual, expected, "recomputed bootstrap bound")
    require(len(results["primary"]["percentile_95_interval_ev"]) == 2, "bootstrap bound count")
    close(results["primary"]["mean_difference_ev"], np.average(paired), "recomputed paired mean")
    generator = np.random.default_rng(20261019)
    orders = [generator.permutation(len(keys)) for _ in range(config["random_replicates"])]
    lo, hi = config["target_window_ev"]
    hits = np.asarray([lo <= u["target"] <= hi for u in units])
    checked = 0
    for summary in results["ranking"]:
        if summary["policy"] != "random":
            continue
        counts = [int(np.count_nonzero(hits[order[:summary["budget"]]])) for order in orders]
        require(counts == summary["replicate_hits"], "random ranking differs from frozen permutation stream")
        close(summary["hits"], sum(counts) / len(counts), "recomputed random hits")
        checked += 1
    return {"bootstrap_interval_ev": interval.tolist(), "bootstrap_resamples": len(samples),
            "random_ranking_cells_checked": checked, "random_replicates": len(orders),
            "numpy_version": np.__version__}


def verify_resampling(root, run_id):
    run = root / "runs" / run_id
    result = audit_resamples(table(run / "predictions/oof.csv"), read(run / "units.json"),
                            read(run / "config.json"), read(run / "results.json"))
    return {"run_id": run_id, "status": "RESAMPLING_VERIFIED", **result,
            "scope": "Independent arithmetic implementation; shared specified NumPy RNG; not independent scientific replication"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify(args.root, args.run_id)
    result = verify_resampling(args.root, args.run_id)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
