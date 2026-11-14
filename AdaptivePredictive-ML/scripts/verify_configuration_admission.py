#!/usr/bin/env python3
"""Verify the input guard repair against the exact unchanged scientific model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from adaptive_predictive.model import DDFSConfig, DriftDirectedFixedShare


BASE = "22461b231d4c44fc41562bf24b128a67478524b3"
RELATIVE = "AdaptivePredictive-ML/src/adaptive_predictive/model.py"
MODES = ("uniform", "hedge", "adahedge", "fixed_share", "adaptive_share", "directed_fixed_share", "ddfs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref", default=BASE)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    original = subprocess.run(["git", "show", f"{args.baseline_ref}:{RELATIVE}"], cwd=repository, capture_output=True, check=True).stdout
    baseline = types.ModuleType("adaptive_predictive_pinned_original")
    sys.modules[baseline.__name__] = baseline
    exec(compile(original, f"{args.baseline_ref}:{RELATIVE}", "exec"), baseline.__dict__)
    configs = [
        {},
        {"forgetting_factors": (0.95,), "lags": (1,), "interval_window": 8},
        {"forgetting_factors": (0.85, 0.95, 0.99, 1.0), "lags": (1, 3, 7), "interval_alpha": 0.2},
    ]
    checked = 0
    trials = []
    rng = np.random.default_rng(20261114)
    for config_index, config in enumerate(configs):
        for season in (1, 4, 7):
            for mode in MODES:
                left = baseline.DriftDirectedFixedShare(season=season, mode=mode, config=baseline.DDFSConfig(**config))
                right = DriftDirectedFixedShare(season=season, mode=mode, config=DDFSConfig(**config))
                warmup = 20 + rng.standard_normal(left.minimum_history)
                for value in warmup:
                    left.observe_initial(value)
                    right.observe_initial(value)
                for step in range(128):
                    a, b = left.predict(), right.predict()
                    if asdict(a) != asdict(b):
                        raise AssertionError("prediction changed under a valid configuration")
                    value = 20 + 3 * np.sin(step / 4) + (10 if step >= 64 else 0) + rng.normal(0, 0.2)
                    if step % 31 == 0:
                        value += 50
                    left.update(value)
                    right.update(value)
                    for before, after in zip(left.experts, right.experts, strict=True):
                        if not np.array_equal(before.theta, after.theta) or not np.array_equal(before.covariance, after.covariance):
                            raise AssertionError("expert state changed under a valid configuration")
                    if not np.array_equal(left.weights, right.weights):
                        raise AssertionError("aggregation weights changed")
                    checked += 1
                trials.append({"config_index": config_index, "season": season, "mode": mode, "forecasts": 128, "prediction_and_expert_state_exact": True})
    invalid_cases = [
        {"ridge": float("nan")}, {"eta": float("nan")}, {"drift_gain": float("nan")},
        {"prediction_clip": float("nan")}, {"loss_clip": float("nan")}, {"min_scale": float("nan")},
        {"interval_window": 0}, {"interval_window": -1}, {"interval_window": 2.5},
        {"lags": ()}, {"lags": (1.5,)}, {"lags": (True,)}, {"min_scale": -1.0},
        {"prediction_clip": -1.0}, {"loss_clip": 0.0}, {"drift_gain": -1.0},
    ]
    admitted_before, rejected_after = 0, 0
    for config in invalid_cases:
        try:
            baseline.DriftDirectedFixedShare(season=4, config=baseline.DDFSConfig(**config))
            admitted_before += 1
        except (ValueError, TypeError):
            pass
        try:
            DriftDirectedFixedShare(season=4, config=DDFSConfig(**config))
        except ValueError:
            rejected_after += 1
        else:
            raise AssertionError("invalid configuration still admitted")
    output = {
        "status": "VERIFIED_CONFIGURATION_ADMISSION_REPAIR",
        "baseline_commit": args.baseline_ref,
        "baseline_source_sha256": hashlib.sha256(original).hexdigest(),
        "repaired_source_sha256": hashlib.sha256((repository/RELATIVE).read_bytes()).hexdigest(),
        "synthetic_forecasts_compared": checked,
        "configuration_mode_season_trials": len(trials),
        "mismatches": 0,
        "invalid_cases": len(invalid_cases),
        "invalid_cases_admitted_before": admitted_before,
        "invalid_cases_rejected_after": rejected_after,
        "new_efficacy_claim": False,
        "protected_data_access": False,
        "scope": "Exact regression check on bounded synthetic sequences. This is configuration hardening and does not change or validate the historical predictive advantage claim.",
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in output.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
