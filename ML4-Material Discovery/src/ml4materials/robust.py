"""Versioned, group-isolated robust-aggregation development experiment.

No functions in this module change historical v1/v2 training or artifacts.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from .chemistry import element_fractions
from .data import load_data_contract, load_json, load_records, sha256_file, write_json
from .evaluation import interval_metrics, regression_metrics
from .experiment import _utc_now, _write_csv, create_artifact_manifest
from .features import CompositionFeaturizer
from .guards import ResourceGuard, ResourceLimitError
from .models import fit_ensemble, make_ridge
from .uncertainty import normal_target_probability


def canonical_units(records) -> list[dict[str, Any]]:
    grouped: dict[str, list] = {}
    for record in records:
        grouped.setdefault(record.canonical_composition, []).append(record)
    return [{"canonical": key, "composition": rows[0].composition,
             "target": float(np.mean([r.gap_ev for r in rows])),
             "source_ids": [r.material_id for r in rows],
             "source_targets": [r.gap_ev for r in rows]}
            for key, rows in sorted(grouped.items())]


def hash_order(keys: list[str], seed: int) -> np.ndarray:
    if len(set(keys)) != len(keys):
        raise ValueError("canonical units must be unique")
    return np.asarray(sorted(range(len(keys)), key=lambda i: hashlib.sha256(
        f"{seed}:{keys[i]}".encode()).hexdigest()), dtype=int)


def grouped_folds(keys: list[str], seed: int, n_folds: int = 5) -> np.ndarray:
    if n_folds < 2 or len(keys) < n_folds:
        raise ValueError("need at least two nonempty folds")
    folds = np.empty(len(keys), dtype=int)
    folds[hash_order(keys, seed)] = np.arange(len(keys)) % n_folds
    return folds


def proper_calibration(keys: list[str], outer: np.ndarray, seed: int, fraction: float):
    if not 0 < fraction < 0.5 or len(outer) < 3:
        raise ValueError("invalid calibration split")
    order = outer[hash_order([keys[i] for i in outer], seed)]
    n_cal = max(1, int(round(len(order) * fraction)))
    return np.sort(order[n_cal:]), np.sort(order[:n_cal])


def tree_matrix(models, features: np.ndarray) -> np.ndarray:
    return np.column_stack([tree.predict(features).astype(np.float32)
                            for model in models for tree in model.estimators_])


def aggregate(matrix: np.ndarray, method: str) -> np.ndarray:
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("finite nonempty tree predictions required")
    if method == "mean":
        return matrix.mean(axis=1, dtype=np.float64)
    if method == "median":
        return np.median(matrix.astype(np.float64), axis=1)
    raise ValueError(f"unknown aggregation {method}")


def conformal_radius(scores: np.ndarray, coverage: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("finite nonnegative calibration scores required")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie in (0, 1)")
    rank = int(np.ceil((len(scores) + 1) * coverage))
    return float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else float("inf")


def paired_bootstrap(differences: np.ndarray, repetitions: int, seed: int) -> dict:
    """Resample units, never treat CV folds/repeats as independent data."""
    if differences.ndim != 2 or not np.isfinite(differences).all():
        raise ValueError("complete repeat-by-unit error differences required")
    per_unit = differences.mean(axis=0)
    rng = np.random.default_rng(seed)
    estimates = np.asarray([per_unit[rng.integers(0, len(per_unit), len(per_unit))].mean()
                            for _ in range(repetitions)])
    return {"mean_difference_ev": float(per_unit.mean()),
            "percentile_95_interval_ev": np.quantile(estimates, [.025, .975]).tolist(),
            "resamples": repetitions, "unit": "canonical_composition",
            "scope": "exploratory conditional-on-fitted-models; not independent-CV or confirmatory inference"}


def source_identity(root: Path) -> dict:
    paths = sorted([*root.glob("src/ml4materials/*.py"), *root.glob("tests/*.py"),
                    *root.glob("scripts/*.py"), *root.glob("configs/**/*.json"),
                    root / "PROTOCOLS/robust-aggregation-v3.md", root / "pyproject.toml"])
    files = {str(p.relative_to(root)): sha256_file(p) for p in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def quantile_collapse_diagnostic() -> dict:
    """Hand-interpretable zero-mass fixture; no benchmark retuning."""
    x = np.arange(200, dtype=float).reshape(-1, 1) / 199
    y = np.where(x[:, 0] < .6, 0., 1. + x[:, 0])
    reports = {}
    for name, init in [("default_median", None), ("positive_initialization", DummyRegressor(strategy="constant", constant=.1))]:
        model = GradientBoostingRegressor(loss="quantile", alpha=.5, n_estimators=40,
                                         max_depth=2, learning_rate=.1, random_state=7, init=init).fit(x, y)
        initial = model.init_.predict(x)
        gradient = model._loss.gradient(y, initial)
        reports[name] = {"initial_prediction": float(initial[0]),
                         "unique_initial_gradients": np.unique(gradient).tolist(),
                         "first_tree_nodes": int(model.estimators_[0, 0].tree_.node_count),
                         "unique_final_predictions": int(len(np.unique(model.predict(x)))),
                         "training_mae_ev": float(np.mean(np.abs(y - model.predict(x))))}
    return {"fixture": "200 ordered units; 60% zero, 40% positive deterministic response",
            "models": reports, "interpretation": "synthetic mechanism diagnostic; no new benchmark performance claim"}


def _ranking(units, target, arrays, config, repeat):
    window = config["target_window_ev"]
    hit = (target >= window[0]) & (target <= window[1])
    median, sigma, mass = arrays["et_median"], arrays["sigma"], arrays["tree_mass"]
    gaussian = normal_target_probability(median, np.maximum(sigma, config["sigma_floor_ev"]), *window)
    scores = {"empirical_tree_mass": mass, "gaussian_working": gaussian,
              "greedy_median": -np.abs(median - np.mean(window))}
    rows, ranks = [], {}
    for name, score in scores.items():
        order = np.argsort(-score, kind="stable")
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(len(order)) + 1
        ranks[name] = rank
        for budget in config["ranking_budgets"]:
            if budget > len(target):
                continue
            hits = int(hit[order[:budget]].sum())
            rows.append({"repeat": repeat, "policy": name, "budget": budget, "hits": hits,
                         "precision": hits / budget, "enrichment": hits / budget / hit.mean()})
    rng = np.random.default_rng(20261019)
    orders = [rng.permutation(len(target)) for _ in range(config["random_replicates"])]
    for budget in config["ranking_budgets"]:
        if budget <= len(target):
            counts = [int(hit[order[:budget]].sum()) for order in orders]
            rows.append({"repeat": repeat, "policy": "random", "budget": budget,
                         "hits": float(np.mean(counts)), "precision": float(np.mean(counts) / budget),
                         "enrichment": float(np.mean(counts) / budget / hit.mean()), "replicate_hits": counts})
    ledger = [{"repeat": repeat, "canonical": u["canonical"], "target": float(target[i]),
               "prediction": float(median[i]), "sigma": float(sigma[i]),
               **{name: float(score[i]) for name, score in scores.items()},
               **{name + "_rank": int(rank[i]) for name, rank in ranks.items()}}
              for i, u in enumerate(units)]
    return rows, ledger


def predict_robust(checkpoint_path: Path, formulas: list[str]) -> list[dict]:
    payload = joblib.load(checkpoint_path)
    if payload.get("schema") != "robust-aggregation-v3:1":
        raise ValueError("unsupported robust checkpoint schema")
    featurizer = payload["featurizer"]
    if list(featurizer.feature_names) != payload["feature_names"]:
        raise ValueError("feature schema mismatch")
    matrix = tree_matrix(payload["models"], featurizer.transform(formulas))
    prediction = aggregate(matrix, "median")
    scale = np.maximum(matrix.std(axis=1, ddof=1, dtype=np.float64), payload["sigma_floor_ev"])
    radius = conformal_radius(np.asarray(payload["scores"]["median_normalized"]), .9)
    return [{"composition": f, "prediction_ev": float(prediction[i]),
             "lower_90_ev": float(prediction[i] - radius * scale[i]),
             "upper_90_ev": float(prediction[i] + radius * scale[i]),
             "unseen_elements": list(featurizer.unseen_elements(f)),
             "scope": "single development fold; not a discovery or OOD validation"}
            for i, f in enumerate(formulas)]


def run_robust(root: Path, run_id: str, config_path: Path) -> Path:
    root = root.resolve()
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run ID must be one safe path component")
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"immutable run exists: {run_dir}")
    config = load_json(config_path)
    if config["study_version"] != "robust-aggregation-v3":
        raise ValueError("wrong study configuration")
    run_dir.mkdir(parents=True)
    started = time.monotonic()
    receipt = {"run_id": run_id, "started_utc": _utc_now(), "status": "RUNNING",
               "source_identity": source_identity(root), "config_source_sha256": sha256_file(config_path),
               "python": platform.python_version(), "platform": platform.platform(),
               "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "joblib"]}}
    write_json(run_dir / "config.json", config)
    receipt["config_sha256"] = sha256_file(run_dir / "config.json")
    write_json(run_dir / "receipt.json", receipt)
    failures, events = [], []
    guard = None
    try:
        guard = ResourceGuard(float(config["max_rss_mib"]), config["require_no_pageout_increase"])
        if config["require_no_pageout_increase"] and guard.starting_pageouts is None:
            raise ResourceLimitError("pageout monitor unavailable")
        def check(phase):
            if time.monotonic() - started > config["max_wall_seconds"]:
                raise ResourceLimitError("prespecified wall-time budget exceeded")
            event = {"utc": _utc_now(), **guard.check(phase)}
            events.append(event)
            write_json(run_dir / "events.json", events)
            print(json.dumps(event), flush=True)

        contract = load_data_contract(root, config["raw_data_contract"])
        units = canonical_units(load_records(root, contract))
        receipt["dataset_sha256"] = contract["source_sha256"]
        write_json(run_dir / "units.json", units)
        keys, formulas = [u["canonical"] for u in units], [u["composition"] for u in units]
        target = np.asarray([u["target"] for u in units])
        n = len(units)
        diagnostics = quantile_collapse_diagnostic()
        write_json(run_dir / "quantile_diagnostic.json", diagnostics)
        check("data_and_diagnostic")
        prediction_rows, metric_rows, calibration_rows, fold_rows = [], [], [], []
        ranking_rows, ranking_ledger, subgroup_rows, probes, splits = [], [], [], [], []
        error_differences = []
        model_names = ["et_mean", "et_median", "et_median_96", "et_fractions_median", "random_forest", "hist_gradient_boosting", "ridge", "train_median"]

        def ensemble_cell(fit, cal, test, name, save=True):
            featurizer = CompositionFeaturizer().fit([formulas[i] for i in fit])
            xf, xc, xt = [featurizer.transform([formulas[i] for i in indices]) for indices in [fit, cal, test]]
            t0 = time.monotonic()
            models = fit_ensemble(xf, target[fit], config["model_seeds"], config["extra_trees"])
            cm, tm = tree_matrix(models, xc), tree_matrix(models, xt)
            sigma_c = np.maximum(cm.std(axis=1, ddof=1, dtype=np.float64), config["sigma_floor_ev"])
            sigma_t = np.maximum(tm.std(axis=1, ddof=1, dtype=np.float64), config["sigma_floor_ev"])
            output = {"et_mean": aggregate(tm, "mean"), "et_median": aggregate(tm, "median"),
                      "et_median_96": aggregate(tm[:, :config["extra_trees"]["n_estimators_per_seed"]], "median"),
                      "sigma": sigma_t,
                      "tree_mass": np.mean((tm >= config["target_window_ev"][0]) & (tm <= config["target_window_ev"][1]), axis=1)}
            scores, intervals = {}, {}
            for method in ["mean", "median"]:
                residual = np.abs(target[cal] - aggregate(cm, method))
                for scale_name, sc, st in [("normalized", sigma_c, sigma_t), ("constant", np.ones(len(cal)), np.ones(len(test)))]:
                    key = method + "_" + scale_name
                    scores[key] = (residual / sc).tolist()
                    for coverage in config["interval_coverages"]:
                        radius = conformal_radius(residual / sc, coverage)
                        center = output["et_" + method]
                        intervals[(key, coverage)] = (center - radius * st, center + radius * st)
            fit_seconds = time.monotonic() - t0
            node_count = sum(t.tree_.node_count for m in models for t in m.estimators_)
            if save:
                path = run_dir / "models" / (name + ".joblib")
                path.parent.mkdir(exist_ok=True)
                payload = {"schema": "robust-aggregation-v3:1", "models": models, "featurizer": featurizer,
                           "feature_names": list(featurizer.feature_names), "scores": scores,
                           "sigma_floor_ev": config["sigma_floor_ev"], "fit_keys": [keys[i] for i in fit],
                           "calibration_keys": [keys[i] for i in cal], "config_sha256": receipt["config_sha256"]}
                temporary = path.with_suffix(".tmp")
                joblib.dump(payload, temporary, compress=3)
                temporary.replace(path)
                del payload, models
                gc.collect()
                reloaded = joblib.load(path)
                reloaded_matrix = tree_matrix(reloaded["models"], xt)
                if not np.array_equal(tm, reloaded_matrix):
                    raise RuntimeError("checkpoint changed tree predictions")
                del reloaded, reloaded_matrix
            else:
                del models
            del cm, tm
            gc.collect()
            check(name + "_ensemble")
            return output, intervals, xf, xt, fit_seconds, node_count

        for repeat, seed in enumerate(config["split_seeds"]):
            folds = grouped_folds(keys, seed, config["n_folds"])
            arrays = {name: np.full(n, np.nan) for name in model_names + ["sigma", "tree_mass"]}
            bounds = {(method + "_" + scale, c): (np.full(n, np.nan), np.full(n, np.nan))
                      for method in ["mean", "median"] for scale in ["normalized", "constant"] for c in config["interval_coverages"]}
            for fold in range(config["n_folds"]):
                outer, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                fit, cal = proper_calibration(keys, outer, seed + 731, config["calibration_fraction"])
                name = f"repeat_{repeat}_fold_{fold}"
                splits.append({"cell": name, "seed": seed, "fit": fit.tolist(), "calibration": cal.tolist(), "test": test.tolist()})
                write_json(run_dir / "splits.json", splits)
                output, intervals, xf, xt, duration, nodes = ensemble_cell(fit, cal, test, name)
                fold_rows.append({"cell": name, "model": "shared_extra_trees", "fit_n": len(fit), "test_n": len(test),
                                  "fit_predict_seconds": duration, "tree_nodes": nodes,
                                  "mean_mae_ev": regression_metrics(target[test], output["et_mean"])["mae_ev"],
                                  "median_mae_ev": regression_metrics(target[test], output["et_median"])["mae_ev"]})
                for key, values in output.items():
                    arrays[key][test] = values
                for key, (lower, upper) in intervals.items():
                    bounds[key][0][test], bounds[key][1][test] = lower, upper
                t0 = time.monotonic()
                fraction_models = fit_ensemble(xf[:, :118], target[fit], config["model_seeds"], config["extra_trees"])
                arrays["et_fractions_median"][test] = aggregate(tree_matrix(fraction_models, xt[:, :118]), "median")
                fold_rows.append({"cell": name, "model": "fractions_only", "fit_predict_seconds": time.monotonic() - t0})
                del fraction_models
                gc.collect()
                check(name + "_fractions_ablation")
                baselines = {
                    "random_forest": RandomForestRegressor(n_estimators=288, max_depth=22, min_samples_leaf=2, max_features=.7, n_jobs=1, random_state=7),
                    "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=160, learning_rate=.05, max_leaf_nodes=31, min_samples_leaf=20, early_stopping=False, random_state=7),
                    "ridge": make_ridge([.001, .01, .1, 1, 10, 100]),
                    "train_median": DummyRegressor(strategy="median")}
                for model_name in list(baselines):
                    model = baselines.pop(model_name)
                    t0 = time.monotonic()
                    model.fit(xf, target[fit])
                    arrays[model_name][test] = model.predict(xt)
                    fold_rows.append({"cell": name, "model": model_name, "fit_predict_seconds": time.monotonic() - t0,
                                      "mae_ev": regression_metrics(target[test], arrays[model_name][test])["mae_ev"]})
                    del model
                    gc.collect()
                    check(name + "_" + model_name)
                if repeat == 0 and fold == 0:
                    rng = np.random.default_rng(731)
                    order = rng.permutation(len(fit))
                    for probe in ["scale_25", "scale_50", "noise_0.1", "permuted"]:
                        fraction = {"scale_25": .25, "scale_50": .5}.get(probe, 1.)
                        selected = order[:max(4, int(len(fit) * fraction))]
                        yfit = target[fit[selected]].copy()
                        if probe == "noise_0.1":
                            yfit += np.random.default_rng(20261021).normal(0, .1, len(yfit))
                        if probe == "permuted":
                            yfit = np.random.default_rng(101).permutation(yfit)
                        t0 = time.monotonic()
                        models = fit_ensemble(xf[selected], yfit, [7], config["extra_trees"])
                        prediction = aggregate(tree_matrix(models, xt), "median")
                        probes.append({"probe": probe, "cell": name, "fit_n": len(selected),
                                       "duration_seconds": time.monotonic() - t0, **regression_metrics(target[test], prediction)})
                        del models
                        gc.collect()
                        check(name + "_" + probe)
                del xf, xt
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise RuntimeError("incomplete outer predictions")
            error_differences.append(np.abs(target - arrays["et_median"]) - np.abs(target - arrays["et_mean"]))
            for model_name in model_names:
                metric_rows.append({"repeat": repeat, "seed": seed, "model": model_name, **regression_metrics(target, arrays[model_name])})
                masks = {"zero": target == 0, "positive": target > 0,
                         "contains_oxygen": np.asarray(["O" in element_fractions(f) for f in formulas]),
                         "conflicting_duplicates": np.asarray([len(u["source_ids"]) > 1 for u in units])}
                for subgroup, mask in masks.items():
                    if mask.any():
                        subgroup_rows.append({"repeat": repeat, "model": model_name, "subgroup": subgroup, "n": int(mask.sum()), **regression_metrics(target[mask], arrays[model_name][mask])})
            for (method, coverage), (lower, upper) in bounds.items():
                calibration_rows.append({"repeat": repeat, "method": method, **interval_metrics(target, lower, upper, coverage)})
            ranked, ledger = _ranking(units, target, arrays, config, repeat)
            ranking_rows.extend(ranked)
            ranking_ledger.extend(ledger)
            for i, unit in enumerate(units):
                prediction_rows.append({"repeat": repeat, "fold": int(folds[i]), "canonical": unit["canonical"], "target": float(target[i]),
                                        **{key: float(values[i]) for key, values in arrays.items()},
                                        **{f"{method}_{int(c * 100)}_{side}": float(values[i])
                                           for (method, c), pair in bounds.items() for side, values in zip(["lower", "upper"], pair)}})
            _write_csv(run_dir / "predictions/oof.csv", prediction_rows)
            _write_csv(run_dir / "tables/metrics.csv", metric_rows)
            check(f"repeat_{repeat}_completed")
        holdout_rows, holdout_predictions = [], []
        for element in config["holdout_elements"]:
            mask = np.asarray([element in element_fractions(f) for f in formulas])
            outer, test = np.flatnonzero(~mask), np.flatnonzero(mask)
            fit, cal = proper_calibration(keys, outer, 731, config["calibration_fraction"])
            name = "holdout_" + element
            splits.append({"cell": name, "fit": fit.tolist(), "calibration": cal.tolist(), "test": test.tolist()})
            output, intervals, xf, xt, duration, nodes = ensemble_cell(fit, cal, test, name)
            for method in ["mean", "median"]:
                lo, hi = intervals[(method + "_normalized", .9)]
                holdout_rows.append({"element": element, "model": "et_" + method, "n": len(test),
                                     "duration_shared_seconds": duration, **regression_metrics(target[test], output["et_" + method]),
                                     **interval_metrics(target[test], lo, hi, .9)})
                holdout_predictions.extend({"element": element, "model": "et_" + method, "canonical": keys[index],
                                             "target": float(target[index]), "prediction": float(output["et_" + method][j]),
                                             "lower": float(lo[j]), "upper": float(hi[j])} for j, index in enumerate(test))
            del xf, xt
        differences = np.asarray(error_differences)
        per_repeat = differences.mean(axis=1)
        threshold = config["minimum_mae_improvement_ev"]
        disposition = "SUPPORTED" if np.all(per_repeat <= -threshold) else ("NEGATIVE" if np.all(per_repeat >= 0) else "MIXED")
        results = {"study_version": config["study_version"], "evaluation_class": config["evaluation_class"],
                   "disposition": disposition, "canonical_units": n, "cells_completed": len(splits),
                   "primary": {"median_minus_mean_mae_ev_by_repeat": per_repeat.tolist(), "required_reduction_ev": threshold,
                               **paired_bootstrap(differences, config["bootstrap_repetitions"], 20261021)},
                   "metrics": metric_rows, "calibration": calibration_rows, "ranking": ranking_rows,
                   "holdouts": holdout_rows, "probes": probes, "checkpoint_reload": "EXACT_ALL_CELLS"}
        for relative, rows in [("tables/calibration.csv", calibration_rows), ("tables/ranking.csv", ranking_rows),
                               ("tables/holdouts.csv", holdout_rows), ("tables/subgroups.csv", subgroup_rows),
                               ("tables/costs.csv", fold_rows), ("tables/probes.csv", probes),
                               ("predictions/ranking.csv", ranking_ledger), ("predictions/holdouts.csv", holdout_predictions)]:
            _write_csv(run_dir / relative, rows)
        write_json(run_dir / "splits.json", splits)
        write_json(run_dir / "results.json", results)
        check("complete")
        receipt["status"] = "COMPLETED"
    except Exception as exc:
        receipt["status"] = "FAILED_RETAINED"
        failures.append({"utc": _utc_now(), "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    finally:
        receipt.update({"finished_utc": _utc_now(), "duration_seconds": time.monotonic() - started})
        if guard is not None:
            receipt["resources"] = guard.receipt()
        write_json(run_dir / "receipt.json", receipt)
        write_json(run_dir / "failures.json", failures)
        write_json(run_dir / "events.json", events)
        create_artifact_manifest(run_dir)
    if receipt["status"] != "COMPLETED":
        raise RuntimeError(f"retained failed run: {run_dir}; {failures[-1]['message']}")
    return run_dir
