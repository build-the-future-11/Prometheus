"""Bounded synthetic end-to-end verification, separate from scientific evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from unittest.mock import patch

from ml4materials.chemistry import canonical_composition
from ml4materials.data import MaterialRecord, write_json
from ml4materials.robust import run_robust, predict_robust
from ml4materials.experiment import verify_artifact_manifest


def main():
    # 200 distinct binary compositions, all named holdouts represented.
    formulas = [f"{element}{i}O" for element in ["Li", "Zn", "Ga", "Se", "Te"] for i in range(1, 41)]
    records = [MaterialRecord(str(i), f, float(i % 7) / 3, canonical_composition(f)) for i, f in enumerate(formulas)]
    original = Path(__file__).resolve().parents[1]
    config = json.loads((original / "configs/studies/robust-aggregation-v3.json").read_text())
    config.update(split_seeds=[20261021], n_folds=2, bootstrap_repetitions=20)
    config["extra_trees"].update(n_estimators_per_seed=4, max_depth=4)
    with TemporaryDirectory(prefix="ml4-dry-") as temporary:
        root = Path(temporary)
        write_json(root / "config.json", config)
        write_json(root / "configs/data_contract.json", {"source_sha256": "synthetic"})
        # source_identity requires real source paths, so use its captured original value.
        from ml4materials.robust import source_identity
        identity = source_identity(original)
        with patch("ml4materials.robust.load_records", return_value=records), patch("ml4materials.robust.source_identity", return_value=identity):
            run = run_robust(root, "synthetic-dry-run", root / "config.json")
        assert verify_artifact_manifest(run)["ok"]
        predictions = predict_robust(run / "models/repeat_0_fold_0.joblib", ["Li2O", "O2Li4"])
        assert predictions[0]["prediction_ev"] == predictions[1]["prediction_ev"]
        result = json.loads((run / "results.json").read_text())
        assert result["cells_completed"] == 7
        print(json.dumps({"status": "SYNTHETIC_E2E_PASS", "cells": 7, "scope": "software verification only"}))


if __name__ == "__main__":
    main()
