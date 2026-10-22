import copy
import json
import tempfile
import unittest
from pathlib import Path

from piml.experiment import run, validate_config, verify
from piml.report import generate


BASE_CONFIG = {
    "study_version": "unit-v1",
    "phase": "development",
    "frozen": False,
    "grid_points": 17,
    "modes": 3,
    "data": {
        "common": {"coefficient_scale": 0.45, "source_scale": 1.0},
        "train": {"seed": 1000, "count": 20},
        "calibration": {"seed": 2000, "count": 6},
        "iid_test": {"seed": 3000, "count": 5},
        "ood_test": {"seed": 4000, "count": 5, "coefficient_scale": 0.9},
    },
    "model": {"ridge": 0.0001, "ensemble_members": 2, "bootstrap_seed": 77},
    "solver": {"rtol": 1e-9, "atol": 1e-12, "max_iterations": 15},
    "controls": {"jacobi_steps": 2, "jacobi_relaxation": 0.8},
    "statistics": {"bootstrap_seed": 99, "bootstrap_repeats": 200},
    "decision": {"primary_split": "iid_test", "minimum_mean_reduction": 0.1, "minimum_ci_lower": -1.0},
}


class ExperimentTests(unittest.TestCase):
    def test_overlapping_seed_ranges_are_rejected(self) -> None:
        config = copy.deepcopy(BASE_CONFIG)
        config["data"]["calibration"]["seed"] = 1010
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_config(config)

    def test_unfrozen_confirmatory_config_is_rejected(self) -> None:
        config = copy.deepcopy(BASE_CONFIG)
        config["phase"] = "confirmatory"
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_config(config)

    def test_end_to_end_run_verify_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run(copy.deepcopy(BASE_CONFIG), root / "run")
            self.assertEqual(result["verification"]["status"], "PASS")
            analysis = json.loads((root / "run" / "analysis.json").read_text())
            self.assertEqual(set(analysis["paired_primary"]), {"iid_test", "ood_test"})
            report = generate(root / "run", root / "paper")
            self.assertEqual(report["verification"], "PASS")
            self.assertTrue((root / "paper" / "results_table.tex").is_file())

    def test_manifest_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run(copy.deepcopy(BASE_CONFIG), run_dir)
            with (run_dir / "metrics.csv").open("a") as handle:
                handle.write("tampered\n")
            self.assertEqual(verify(run_dir)["status"], "FAIL")

    def test_nonempty_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run(copy.deepcopy(BASE_CONFIG), run_dir)
            with self.assertRaises(FileExistsError):
                run(copy.deepcopy(BASE_CONFIG), run_dir)


if __name__ == "__main__":
    unittest.main()

