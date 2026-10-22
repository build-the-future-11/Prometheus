"""Regression tests for the read-only v1 publication analysis pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_bundle", ROOT / "scripts" / "analyze_bundle.py")
assert SPEC and SPEC.loader
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


class AnalyzeBundleTests(unittest.TestCase):
    def test_v1_analysis_recomputes_verified_baseline_summary(self) -> None:
        bundle = ROOT / "results" / "development" / "shared-baselines-v1-20261019"
        analysis = ANALYZE.build_analysis(bundle)
        self.assertEqual(analysis["run_accounting"]["completed"], 20)
        self.assertEqual(analysis["run_accounting"]["protected_evaluation_runs"], 0)
        self.assertAlmostEqual(analysis["policy_summaries"]["sgd_lr_0p01"]["primary_mse_mean"], 0.9484112278453154)
        self.assertEqual(len(analysis["rolling_mse_curves"]["sgd_lr_0p01"]), 480)
        self.assertEqual(len(analysis["attempt_rolling_mse_curves"]), 20)
        self.assertEqual(analysis["paired_against_no_adaptation"]["sgd_lr_0p01"]["negative_difference_count"], 5)

    def test_output_is_complete_and_never_overwritten(self) -> None:
        bundle = ROOT / "results" / "development" / "shared-baselines-v1-20261019"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            ANALYZE.write_outputs(ANALYZE.build_analysis(bundle), output)
            self.assertTrue((output / "analysis.json").is_file())
            self.assertTrue((output / "figures" / "development_rolling_mse.svg").is_file())
            with self.assertRaises(FileExistsError):
                ANALYZE.write_outputs(ANALYZE.build_analysis(bundle), output)
