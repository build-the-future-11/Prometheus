"""Verify a completed development bundle and summarize its frozen comparison."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from sparking.evidence import verify_manifest


def verify(root: Path, run_id: str) -> dict:
    run = root / "runs" / run_id
    manifest = verify_manifest(run)
    protocol = json.loads((root / "protocols/development-v1.json").read_text())
    receipt = json.loads((run / "receipt.json").read_text())
    result = json.loads((run / "results.json").read_text())
    freeze = json.loads((run / "freeze_candidate.json").read_text())
    assert receipt["status"] == result["status"] == "SUCCESS"
    assert receipt["run_id"] == result["run_id"] == run_id
    assert result["phase"] == "DEVELOPMENT"
    assert result["protected_confirmatory_outcomes_observed"] is False
    observations = receipt["resource_observations"]
    assert observations and observations[-1]["phase"] == "complete"
    assert all(item["peak_rss_mib"] <= protocol["max_rss_mib"] for item in observations)
    assert all(item["pageout_delta"] == 0 for item in observations)
    elements = protocol["development_elements"]
    grid = list(itertools.product(protocol["gate_grid"]["support_quantile"], protocol["gate_grid"]["power"]))
    controls = [name for name in protocol["models"] if name != "sparking"]
    identities = [(name, None, None) for name in controls] + [("sparking", q, p) for q, p in grid]
    expected = {(element, *identity) for element in elements for identity in identities}
    indexed = {}
    for row in result["rows"]:
        key = (row["element"], row["model"], row.get("support_quantile"), row.get("power"))
        assert key not in indexed, f"Duplicate row: {key}"
        indexed[key] = row
        assert row["n"] == row["test_n"] > 0 and row["train_n"] > 0
        for metric in ("mae_ev", "rmse_ev", "median_ae_ev", "bias_ev"):
            assert math.isfinite(row[metric]), (key, metric)
        assert row["rmse_ev"] + 1e-10 >= row["mae_ev"] >= 0
    assert set(indexed) == expected, "Comparison matrix is incomplete or has undeclared rows"
    summaries = []
    for model, quantile, power in identities:
        per_element = {element: indexed[(element, model, quantile, power)]["mae_ev"] for element in elements}
        macro = statistics.mean(per_element.values())
        reported = result["control_macro_mae_ev"][model] if model != "sparking" else next(
            row["macro_mae_ev"] for row in result["candidate_summary"]
            if (row["support_quantile"], row["power"]) == (quantile, power)
        )
        assert math.isclose(macro, reported, abs_tol=1e-12, rel_tol=0)
        deltas = {
            element: indexed[(element, "residual_ungated", None, None)]["mae_ev"] - value
            for element, value in per_element.items()
        }
        summaries.append({"model": model, "support_quantile": quantile, "power": power,
                          "mae_by_element_ev": per_element, "macro_mae_ev": macro,
                          "improvement_over_ungated_ev": statistics.mean(deltas.values()),
                          "improvement_by_element_ev": deltas})
    candidates = [row for row in summaries if row["model"] == "sparking"]
    assert len(result["candidate_summary"]) == len(grid)
    minimum = min(row["macro_mae_ev"] for row in candidates)
    eligible = [row for row in candidates if row["macro_mae_ev"] <= minimum + 0.002]
    selected = min(eligible, key=lambda row: (row["power"], -row["support_quantile"]))
    for field in ("support_quantile", "power"):
        assert selected[field] == result["selected_gate"][field] == freeze[field]
    return {"schema_version": 1, "run_id": run_id, "verified": True,
            "artifact_manifest_verification": manifest,
            "results_sha256": hashlib.sha256((run / "results.json").read_bytes()).hexdigest(),
            "row_count": len(indexed), "development_elements": elements,
            "elapsed_seconds": receipt["elapsed_seconds"],
            "peak_rss_mib": max(item["peak_rss_mib"] for item in observations),
            "pageout_delta": 0, "selected_gate": selected,
            "comparisons": summaries, "scope": "Development selection only; confirmation remains unrun"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = verify(Path(__file__).resolve().parents[1], args.run_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
