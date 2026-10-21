"""Fixed development-only diagnostics; never replaces frozen v1 outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn._loss.loss import AbsoluteError
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from sparking.chemistry import ATOMIC_NUMBER, featurize
from sparking.data import element_holdout, load_data
from sparking.evidence import ResourceGuard, create_manifest, environment_record, verify_manifest, write_json
from sparking.metrics import regression_metrics
from sparking.model import SparkingRegressor, make_hgb


def describe(values):
    values = np.asarray(values)
    return {"min": float(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "max": float(values.max()),
            "std": float(values.std()), "unique": int(np.unique(values).size)}


def trees(model):
    counts = [int(tree.nodes["is_leaf"].sum()) for stage in model._predictors for tree in stage]
    return {"iterations": int(model.n_iter_), "leaf_count_min": min(counts),
            "leaf_count_max": max(counts), "single_leaf_trees": counts.count(1),
            "initial_prediction": float(model._baseline_prediction.ravel()[0])}


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def main():
    root = Path.cwd()
    run = root / "runs/kaggle-diagnostic-v1-r1"
    run.mkdir(exist_ok=False)
    protocol_path = root / "protocols/diagnostic-v1.json"
    protocol = json.loads(protocol_path.read_text())
    reference = json.loads((root / "diagnostic-reference-results.json").read_text())
    assert hashlib.sha256((root / "diagnostic-reference-results.json").read_bytes()).hexdigest() == protocol["reference_results_sha256"]
    write_json(run / "diagnostic_protocol.json", protocol)
    guard = ResourceGuard(protocol["max_rss_mib"], protocol["require_no_rising_pageout"])
    started = time.time()
    status = "FAILED_RETAINED"
    try:
        guard.check("admission")
        assert sklearn.__version__ == protocol["sklearn_version"]
        assert np.__version__ == protocol["numpy_version"]
        formulas, y = load_data(root)
        x = featurize(formulas)
        guard.check("features")
        config = json.loads((root / "protocols/development-v1.json").read_text())["residual_model"]
        results = []
        for element in protocol["development_elements"]:
            train, test = element_holdout(formulas, element)
            xt, xv, yt, yv = x[train], x[test], y[train], y[test]
            model = SparkingRegressor.fit(xt, yt, config)
            base, residual, distance = model.components(xv)
            ungated = np.clip(base + residual, 0, None)
            ridge = np.clip(base, 0, None)
            median = float(np.median(yt))
            arrays = {"row_index": test, "target": yv, "ridge_raw": base,
                      "residual": residual, "ridge": ridge, "ungated": ungated,
                      "distance": distance, "train_loo_distances": model.leave_one_out_distances}
            predictions = {"median": np.full_like(yv, median), "ridge": ridge,
                           "residual_ungated": ungated, "residual_weight_0.5": np.clip(base + .5 * residual, 0, None)}
            direct = make_hgb(config).fit(xt, yt)
            predictions["direct_hgb"] = np.clip(direct.predict(xv), 0, None)
            gradients = AbsoluteError().gradient(yt, np.full_like(yt, median))
            diagnostic = {"element": element, "train_n": len(train), "test_n": len(test),
                          "training_zero_fraction": float(np.mean(yt == 0)), "training_median": median,
                          "absolute_error_initial_gradients": describe(gradients),
                          "direct_absolute_trees": trees(direct),
                          "direct_absolute_train_prediction": describe(direct.predict(xt)),
                          "direct_absolute_test_prediction": describe(predictions["direct_hgb"]),
                          "residual_absolute_trees": trees(model.residual_model)}
            guard.check(f"{element}_original")
            squared = make_hgb(config).set_params(loss="squared_error").fit(xt, yt)
            predictions["direct_squared"] = np.clip(squared.predict(xv), 0, None)
            diagnostic["direct_squared_trees"] = trees(squared)
            diagnostic["direct_squared_test_prediction"] = describe(predictions["direct_squared"])
            guard.check(f"{element}_direct_squared")
            residual_squared = make_hgb(config).set_params(loss="squared_error").fit(xt, yt - model.ridge.predict(xt))
            predictions["residual_squared_ungated"] = np.clip(base + residual_squared.predict(xv), 0, None)
            diagnostic["residual_squared_trees"] = trees(residual_squared)
            guard.check(f"{element}_residual_squared")
            harm = np.abs(yv - ungated) - np.abs(yv - ridge)
            arrays["residual_harm"] = harm
            index = ATOMIC_NUMBER[element] - 1
            scaled = model.support_scaler.transform(xv)
            nn_index = model.neighbours.kneighbors(scaled, n_neighbors=1, return_distance=False).ravel()
            squared_deltas = (scaled - model.neighbours._fit_X[nn_index]) ** 2
            unseen_share = squared_deltas[:, index] / np.maximum(np.sum(squared_deltas, axis=1), 1e-30)
            arrays["unseen_element_squared_distance_share"] = unseen_share
            arrays["unseen_element_fraction"] = xv[:, index]
            diagnostic["geometry"] = {
                "held_out_element_train_variance": float(model.support_scaler.var_[index]),
                "held_out_element_scale": float(model.support_scaler.scale_[index]),
                "held_out_element_fraction": describe(xv[:, index]),
                "held_out_element_squared_distance_share": describe(unseen_share),
                "train_loo_distance": describe(model.leave_one_out_distances), "test_distance": describe(distance),
                "distance_residual_harm_spearman": corr(distance, harm),
                "unseen_fraction_residual_harm_spearman": corr(xv[:, index], harm),
                "residual_harm_mean_ev": float(harm.mean()),
                "residual_helps_fraction": float(np.mean(harm < 0)),
                "residual_magnitude_mean_ev": float(np.mean(np.abs(residual)))}
            gates = []
            for quantile in (.8, .9, .95, .99):
                for power in (1., 2.):
                    name = f"gate_{quantile}_{power}"
                    pred, gate, _ = model.predict(xv, quantile, power)
                    predictions[name] = pred
                    arrays[name + "_weight"] = gate
                    active = gate < 1 - 1e-12
                    gates.append({"support_quantile": quantile, "power": power,
                                  "threshold": model.threshold(quantile), "active_n": int(active.sum()),
                                  "active_fraction": float(active.mean()), "mean_gate": float(gate.mean()),
                                  "mean_abs_prediction_change_ev": float(np.abs(pred-ungated).mean()),
                                  "active_residual_harm_mean_ev": float(harm[active].mean()) if active.any() else None,
                                  **regression_metrics(yv, pred)})
            diagnostic["gates"] = gates
            # Fixed alternative geometry: Euclidean raw fractions and standardized
            # global descriptors, each block divided by sqrt(feature count).
            gs = StandardScaler().fit(xt[:, 118:])
            alt_train = np.hstack([xt[:, :118] / np.sqrt(118), gs.transform(xt[:, 118:]) / np.sqrt(14)])
            alt_test = np.hstack([xv[:, :118] / np.sqrt(118), gs.transform(xv[:, 118:]) / np.sqrt(14)])
            nn = NearestNeighbors(n_neighbors=2).fit(alt_train)
            alt_loo = nn.kneighbors(alt_train)[0][:, 1]
            alt_distance = nn.kneighbors(alt_test, n_neighbors=1)[0].ravel()
            threshold = float(np.quantile(alt_loo, .8))
            alt_gate = (threshold / np.maximum(alt_distance, threshold)) ** 2
            predictions["alternative_geometry_q0.8_p2"] = np.clip(base + alt_gate * residual, 0, None)
            arrays.update({"alternative_distance": alt_distance, "alternative_gate": alt_gate,
                           "alternative_train_loo_distances": alt_loo})
            diagnostic["alternative_geometry"] = {"threshold": threshold, "mean_gate": float(alt_gate.mean()),
                "active_fraction": float(np.mean(alt_gate < 1 - 1e-12)),
                "distance_residual_harm_spearman": corr(alt_distance, harm)}
            metrics = {name: regression_metrics(yv, pred) for name, pred in predictions.items()}
            original_rows = [row for row in reference["rows"] if row["element"] == element]
            for row in original_rows:
                name = row["model"] if row["model"] != "sparking" else f"gate_{row['support_quantile']}_{row['power']}"
                for metric in ("mae_ev", "rmse_ev", "median_ae_ev", "bias_ev"):
                    if not np.isclose(metrics[name][metric], row[metric], rtol=0, atol=1e-9):
                        raise ValueError(f"Original outcome drift: {element}/{name}/{metric}")
            arrays.update({"prediction_" + name: pred for name, pred in predictions.items()})
            np.savez_compressed(run / f"{element}-predictions.npz", **arrays)
            diagnostic["metrics"] = metrics
            diagnostic["original_all_metrics_reproduced_atol_1e-9"] = True
            results.append(diagnostic)
            write_json(run / f"{element}-diagnostics.json", diagnostic)
            guard.check(f"{element}_complete")
            print(f"DIAGNOSTIC_HOLDOUT_COMPLETE={element}", flush=True)
        summary = {"status": "SUCCESS", "phase": "DEVELOPMENT_DIAGNOSTICS", "elements": results,
                   "protected_confirmatory_outcomes_observed": False,
                   "macro_mae_ev": {name: float(np.mean([r["metrics"][name]["mae_ev"] for r in results])) for name in results[0]["metrics"]}}
        write_json(run / "results.json", summary)
        guard.check("complete")
        status = "SUCCESS"
    except BaseException as error:
        write_json(run / "failures.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        write_json(run / "environment.json", {**environment_record(), "numpy": np.__version__, "sklearn": sklearn.__version__})
        write_json(run / "receipt.json", {"status": status, "elapsed_seconds": time.time()-started,
                                         "resource_observations": guard.observations})
        create_manifest(run)
        verify_manifest(run)


if __name__ == "__main__":
    main()
