from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from neurogenesis.aggregate import aggregate_bundle


def _result(attempt_id: str, policy: str, seed: int, status: str, mse: float = 1.0) -> dict:
    return {
        "schema_version": "neurogenesis.result.v1",
        "attempt_id": attempt_id,
        "status": status,
        "task_id": f"task-{seed}",
        "seed": seed,
        "family": "fixture",
        "split": "development",
        "policy_name": policy,
        "policy_kind": policy,
        "metrics": None
        if status == "FAILED"
        else {
            "primary": {"name": "prequential_mean_squared_error", "value": mse},
            "adaptation_switch_mse": mse,
            "retention_return_mse": mse,
            "resource_usage": {
                "wall_seconds": 0.1,
                "peak_state_bytes": 64,
                "parameter_updates": 1,
            },
        },
        "trace_sha256": None,
        "error": None if status == "SUCCEEDED" else {"type": "FixtureError", "message": "kept"},
    }


class AggregationTests(unittest.TestCase):
    def test_failed_attempts_are_retained_in_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            rows = [
                _result("base-1", "no_adaptation", 1, "SUCCEEDED", 2.0),
                _result("candidate-1", "standard_sgd", 1, "SUCCEEDED", 1.0),
                _result("candidate-2", "standard_sgd", 2, "FAILED"),
            ]
            for row in rows:
                attempt = bundle / "attempts" / row["attempt_id"]
                attempt.mkdir(parents=True)
                (attempt / "result.json").write_text(json.dumps(row), encoding="utf-8")
            aggregate = aggregate_bundle(bundle)
        self.assertEqual(aggregate["attempt_count"], 3)
        self.assertEqual(aggregate["failure_count"], 1)
        self.assertEqual(aggregate["policy_summaries"]["standard_sgd"]["attempted"], 2)
        self.assertEqual(aggregate["policy_summaries"]["standard_sgd"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()

