#!/usr/bin/env python3
"""Fresh synthetic coverage study of the corrected conformal order statistic.

The simulations audit the primitive, not the materials predictor. Infinite
intervals are explicitly reported and are not a useful-prediction performance win.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from sparking import metrics


def run(replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    checks = 0
    for n in (2, 5, 8, 9, 19, 20, 99, 100):
        calibration = np.sort(rng.random((replicates, n)), axis=1)
        test = rng.random(replicates)
        for coverage in (0.8, 0.9, 0.95):
            rank = int(np.ceil((n + 1) * coverage))
            legacy_radius = calibration[:, min(rank, n) - 1]
            repaired_radius = np.full(replicates, np.inf) if rank > n else calibration[:, rank - 1]
            for index in range(min(32, replicates)):
                observed = metrics.conformal_radius(calibration[index], coverage)
                if observed != repaired_radius[index]:
                    raise AssertionError("vectorized/reference implementation disagreement")
                checks += 1
            for scale, scenario in ((1.0, "exchangeable_uniform"), (2.0, "shifted_test_scale_x2")):
                legacy = float(np.mean(test * scale <= legacy_radius))
                repaired = float(np.mean(test * scale <= repaired_radius))
                theoretical_legacy = min(rank, n) / ((n + 1) * scale)
                theoretical_repaired = 1.0 if rank > n else rank / ((n + 1) * scale)
                for value, theoretical in ((legacy, theoretical_legacy), (repaired, theoretical_repaired)):
                    se = np.sqrt(theoretical * (1.0 - theoretical) / replicates)
                    if abs(value - theoretical) > 6 * se + 1 / replicates:
                        raise AssertionError("simulation contradicts exact rank probability beyond six Monte Carlo standard errors")
                rows.append({
                    "scenario": scenario,
                    "calibration_size": n,
                    "nominal_coverage": coverage,
                    "required_rank": rank,
                    "legacy_measured_coverage": legacy,
                    "legacy_exact_coverage": theoretical_legacy,
                    "repaired_measured_coverage": repaired,
                    "repaired_exact_coverage": theoretical_repaired,
                    "unbounded_interval_fraction": float(rank > n),
                    "repaired_mean_radius": None if rank > n else float(np.mean(repaired_radius)),
                    "finite_report_policy": "REFUSE_UNBOUNDED" if rank > n else "FINITE_AVAILABLE",
                    "repetitions": replicates,
                })
    return {
        "schema_version": 1,
        "study_status": "EXECUTED_SYNTHETIC_CALIBRATION_AUDIT",
        "seed": seed,
        "repetitions_per_cell": replicates,
        "cells": len(rows),
        "implementation_reference_checks": checks,
        "code_sha256": hashlib.sha256(Path(inspect.getfile(metrics)).read_bytes()).hexdigest(),
        "new_model_fits": 0,
        "protected_outcomes_observed": False,
        "interpretation": "Boundary correction restores the mathematical rank rule. Small-n cases require infinite intervals, which the finite-metric reporting path explicitly refuses. Finite intervals still lose nominal coverage under the simulated distribution shift.",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.replicates < 1000:
        raise ValueError("at least 1000 replicates are required for this audit")
    output = run(args.replicates, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": output["study_status"], "cells": output["cells"], "implementation_reference_checks": output["implementation_reference_checks"], "example": next(row for row in output["rows"] if row["calibration_size"] == 2 and row["nominal_coverage"] == 0.9 and row["scenario"] == "exchangeable_uniform")}, indent=2))


if __name__ == "__main__":
    main()
