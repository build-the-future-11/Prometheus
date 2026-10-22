"""Bounded synthetic development grid.

This script is deliberately separate from confirmatory execution.  It records every
registered configuration and never selects against protected real-data outcomes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from adaptive_predictive.experiment import canonical_json, run_series, synthetic_series
from adaptive_predictive.model import DDFSConfig


CONFIGURATIONS = (
    {"drift_gain": 0.02, "max_share": 0.03, "fast_rate": 0.20},
    {"drift_gain": 0.04, "max_share": 0.05, "fast_rate": 0.20},
    {"drift_gain": 0.08, "max_share": 0.08, "fast_rate": 0.20},
    {"drift_gain": 0.04, "max_share": 0.05, "fast_rate": 0.10},
    {"drift_gain": 0.08, "max_share": 0.08, "fast_rate": 0.10},
    {"drift_gain": 0.12, "max_share": 0.12, "fast_rate": 0.10},
)
SEEDS = (7, 17, 29)


def main() -> None:
    output = Path("artifacts/development-grid-v1.json")
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    base = DDFSConfig()
    records: list[dict[str, object]] = []
    for config_id, overrides in enumerate(CONFIGURATIONS, start=1):
        config = replace(base, **overrides)
        for seed in SEEDS:
            result = run_series(synthetic_series(720, seed), config, evaluate_from=360)
            ddfs = result["metrics"]["ddfs"]["mase"]  # type: ignore[index]
            fixed = result["metrics"]["fixed_share"]["mase"]  # type: ignore[index]
            records.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "config": asdict(config),
                    "ddfs_mase": ddfs,
                    "fixed_share_mase": fixed,
                    "difference": ddfs - fixed,
                }
            )
    aggregates = []
    for config_id in range(1, len(CONFIGURATIONS) + 1):
        subset = [record for record in records if record["config_id"] == config_id]
        differences = [float(record["difference"]) for record in subset]
        aggregates.append(
            {
                "config_id": config_id,
                "mean_difference": sum(differences) / len(differences),
                "wins": sum(value < 0 for value in differences),
                "n": len(differences),
            }
        )
    payload = {
        "stage": "synthetic-development-not-real-world-evidence",
        "budget": {"registered_configurations": len(CONFIGURATIONS), "seeds": list(SEEDS)},
        "records": records,
        "aggregates": aggregates,
    }
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(canonical_json(payload))
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()

