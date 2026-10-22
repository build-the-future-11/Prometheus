from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from neurogenesis.policies import AdamLinearPolicy, load_checkpoint, save_checkpoint
from neurogenesis.contracts import ContractError


class SerializationTests(unittest.TestCase):
    def test_policy_type_drift_is_rejected(self) -> None:
        policy = AdamLinearPolicy(feature_dim=2, learning_rate=0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_checkpoint(policy, path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["policy_type"] = "standard_sgd"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_checkpoint(policy, path)


if __name__ == "__main__":
    unittest.main()

