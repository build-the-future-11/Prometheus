#!/usr/bin/env python3
"""Run a bounded CPU integration check through the production v3 evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

if not os.environ.get("LOKY_MAX_CPU_COUNT", "").isdigit():
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_successor_v3 import evaluate_outer_holdout, select_candidate  # noqa: E402

from sparking.data import load_data  # noqa: E402
from sparking.successor import formula_element_mask, publication_features  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "protocols" / "development-v3.json").read_text())
    formulas, targets = load_data(root)
    oxygen = formula_element_mask(formulas, "O")
    train_indices = np.flatnonzero(~oxygen)[:512]
    test_indices = np.flatnonzero(oxygen)[:128]
    indices = np.concatenate([train_indices, test_indices])
    subset_formulas = [formulas[int(index)] for index in indices]
    subset_targets = targets[indices]
    matrix = publication_features(subset_formulas)

    protocol["development_elements"] = ["O"]
    protocol["frequency_balance"]["exponents"] = [0.0, 0.5]
    protocol["frequency_balance"]["candidate_exponents"] = [0.5]
    protocol["extra_trees"]["n_estimators"] = 32
    protocol["extra_trees"]["seeds"] = protocol["extra_trees"]["seeds"][:2]
    protocol["catboost"]["iterations"] = 20
    protocol["catboost"]["depth"] = 6
    protocol["catboost"]["seeds"] = protocol["catboost"]["seeds"][:1]
    protocol["corrected_hgb"]["max_iter"] = 20

    with tempfile.TemporaryDirectory(prefix="sparking-v3-dry-run-") as directory:
        run_dir = Path(directory)
        rows, diagnostic = evaluate_outer_holdout(
            run_dir,
            subset_formulas,
            subset_targets,
            matrix.values,
            "O",
            protocol,
            "CPU",
        )
        selection = select_candidate(rows, protocol)
        if len(rows) != 8:
            raise ValueError(f"unexpected dry-run model count: {len(rows)}")
        if not all(np.isfinite(float(row["mae_ev"])) for row in rows):
            raise ValueError("dry-run metrics are non-finite")
        if not (run_dir / "O-predictions.npz").is_file():
            raise ValueError("dry-run prediction artifact was not written")
        result = {
            "status": "DRY_RUN_PASSED",
            "scientific_result": False,
            "train_n": diagnostic["train_n"],
            "test_n": diagnostic["test_n"],
            "model_rows": len(rows),
            "selected_candidate": selection["selected"]["model"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
