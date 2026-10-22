import json
import unittest
from pathlib import Path


class ProtocolTests(unittest.TestCase):
    def test_development_and_confirmatory_seed_universes_are_disjoint(self) -> None:
        development = json.loads(Path("configs/development_v1.json").read_text())
        confirmatory = json.loads(Path("configs/confirmatory_v1.json").read_text())
        def universe(config):
            return {
                seed
                for name in ("train", "calibration", "iid_test", "ood_test")
                for seed in range(config["data"][name]["seed"], config["data"][name]["seed"] + config["data"][name]["count"])
            }
        self.assertFalse(universe(development) & universe(confirmatory))

    def test_confirmatory_protocol_is_frozen(self) -> None:
        config = json.loads(Path("configs/confirmatory_v1.json").read_text())
        self.assertEqual(config["phase"], "confirmatory")
        self.assertIs(config["frozen"], True)
        self.assertIn("frozen_at_utc", config)


if __name__ == "__main__":
    unittest.main()

