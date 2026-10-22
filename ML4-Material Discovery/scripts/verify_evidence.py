"""Independent, standard-library numerical audit of retained prediction artifacts.

Does not import the training/evaluation implementation or deserialize checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(Path(path).read_text())


def table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), message + ": nonfinite")
    require(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10),
            f"{message}: {actual} != {expected}")


def metrics(rows, target, prediction):
    residuals = [float(r[target]) - float(r[prediction]) for r in rows]
    require(bool(residuals) and all(math.isfinite(v) for v in residuals), "missing/nonfinite predictions")
    return {"mae_ev": mean(abs(v) for v in residuals), "rmse_ev": math.sqrt(mean(v * v for v in residuals))}


def formula_ratios(formula):
    """Independent stack parser for source-to-canonical ratio verification."""
    tokens = re.findall(r"[A-Z][a-z]*|\d+(?:\.\d*)?|\.\d+|[()]", formula)
    require("".join(tokens) == formula, "unrecognized source formula")
    stack = [{}]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if token == "(":
            stack.append({})
            continue
        if token == ")":
            require(len(stack) > 1 and stack[-1], "invalid source parentheses")
            amounts = stack.pop()
        else:
            require(token[0].isupper(), "unexpected source amount")
            amounts = {token: Fraction(1)}
        factor = Fraction(1)
        if i < len(tokens) and (tokens[i][0].isdigit() or tokens[i][0] == "."):
            factor = Fraction(tokens[i])
            i += 1
        require(factor > 0, "nonpositive source amount")
        for element, value in amounts.items():
            stack[-1][element] = stack[-1].get(element, Fraction(0)) + factor * value
    require(len(stack) == 1 and bool(stack[0]), "unclosed/empty source formula")
    total = sum(stack[0].values())
    return {e: v / total for e, v in stack[0].items()}


def complete_matrix(rows, columns, expected, label):
    actual = [tuple(row[k] for k in columns) for row in rows]
    require(len(actual) == len(expected) and set(actual) == set(expected),
            "incomplete or duplicate " + label)


def primary_summary(primary, differences, config):
    require(len(primary["median_minus_mean_mae_ev_by_repeat"]) == len(differences), "primary repetition count")
    for actual, expected in zip(primary["median_minus_mean_mae_ev_by_repeat"], differences):
        close(actual, expected, "primary effect")
    close(primary["mean_difference_ev"], mean(differences), "primary mean effect")
    close(primary["required_reduction_ev"], config["minimum_mae_improvement_ev"], "primary threshold")
    require(primary["resamples"] == config["bootstrap_repetitions"], "bootstrap repetition count")
    bounds = primary["percentile_95_interval_ev"]
    require(len(bounds) == 2 and all(math.isfinite(v) for v in bounds) and bounds[0] <= bounds[1], "invalid bootstrap bounds")


def interval(rows, target, lower, upper, coverage):
    triples = [(float(r[target]), float(r[lower]), float(r[upper])) for r in rows]
    require(all(math.isfinite(x) for t in triples for x in t), "nonfinite intervals")
    require(all(lo <= hi for y, lo, hi in triples), "reversed intervals")
    return {"observed_coverage": mean(lo <= y <= hi for y, lo, hi in triples),
            "mean_interval_width_ev": mean(hi - lo for y, lo, hi in triples),
            "mean_interval_score_ev": mean(hi - lo + 2 / (1 - coverage) * (max(lo-y, 0) + max(y-hi, 0)) for y, lo, hi in triples)}


def manifest(run):
    value = read(run / "artifact_manifest.json")
    expected = value["files"]
    observed = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}
    require(observed == set(expected), "manifest file set mismatch")
    for name, info in expected.items():
        path = run / name
        require(path.resolve().is_relative_to(run.resolve()), "unsafe manifest path")
        require(path.stat().st_size == info["bytes"] and digest(path) == info["sha256"], "artifact mismatch: " + name)
    return len(expected)


def verify(root: Path, run_id: str) -> dict:
    run = root / "runs" / run_id
    require(run.resolve().is_relative_to((root / "runs").resolve()), "unsafe run path")
    count = manifest(run)
    receipt = read(run / "receipt.json")
    require(receipt["status"] == "COMPLETED", "run did not complete")
    resources = receipt["resources"]
    require(resources["pageout_monitor_available"], "pageout monitor absent")
    require(resources["pageout_delta"] == 0, "pageout guard violated")
    require(resources["peak_observed_rss_mib"] <= resources["max_allowed_rss_mib"], "RSS ceiling violated")
    require(read(run / "failures.json") == [], "completed run contains failures")
    contract = read(root / "configs/data_contract.json")
    require(digest(root / contract["source_path"]) == contract["source_sha256"], "source dataset hash mismatch")
    raw = json.loads(gzip.decompress((root / contract["source_path"]).read_bytes()))["data"]
    labels = {f"mb-expt-gap-{i+1:04d}": (f, float(y)) for i, (f, y) in enumerate(raw)}
    results = read(run / "results.json")
    recomputed = {}
    if results["study_version"] == "robust-aggregation-v3":
        units, config, splits = read(run / "units.json"), read(run / "config.json"), read(run / "splits.json")
        require(digest(run / "config.json") == receipt["config_sha256"], "config hash mismatch")
        require(receipt["dataset_sha256"] == contract["source_sha256"], "receipt dataset mismatch")
        config_source = root / "configs/studies/robust-aggregation-v3.json"
        freeze = read(root / "artifacts/freeze-v3.json")
        require(digest(config_source) == freeze["files"]["configs/studies/robust-aggregation-v3.json"] == receipt["config_source_sha256"], "frozen configuration source mismatch")
        require(config == read(config_source), "run configuration differs from frozen specification")
        require(receipt["duration_seconds"] <= config["max_wall_seconds"], "wall-time ceiling violated")
        require(resources["max_allowed_rss_mib"] == config["max_rss_mib"], "resource ceiling differs from protocol")
        keys = [u["canonical"] for u in units]
        require(len(set(keys)) == len(keys), "duplicate canonical units")
        require(keys == sorted(keys), "canonical unit order differs from protocol")
        require(results["canonical_units"] == len(keys), "canonical unit count mismatch")
        source_ids = [identifier for u in units for identifier in u["source_ids"]]
        require(len(source_ids) == len(set(source_ids)) == len(labels) and set(source_ids) == set(labels), "source row omissions/duplicates")
        targets = {}
        for u in units:
            amounts = {e: Fraction(v) for e, v in (part.split(":") for part in u["canonical"].split("|"))}
            total = sum(amounts.values())
            ratios = {e: v / total for e, v in amounts.items()}
            require(all(formula_ratios(labels[i][0]) == ratios for i in u["source_ids"]), "source formula/canonical identity mismatch")
            close(u["target"], mean(labels[i][1] for i in u["source_ids"]), "canonical target")
            require(u["source_targets"] == [labels[i][1] for i in u["source_ids"]], "source target mismatch")
            targets[u["canonical"]] = u["target"]
        rows = table(run / "predictions/oof.csv")
        n = len(units)
        require(len(rows) == n * len(config["split_seeds"]), "incomplete repeated predictions")
        require(len({(r["repeat"], r["canonical"]) for r in rows}) == len(rows), "duplicate predictions")
        for r in rows:
            close(r["target"], targets[r["canonical"]], "prediction target")
        expected_cells = {f"repeat_{repeat}_fold_{fold}" for repeat in range(len(config["split_seeds"])) for fold in range(config["n_folds"])} | {"holdout_" + e for e in config["holdout_elements"]}
        require({s["cell"] for s in splits} == expected_cells and len(splits) == len(expected_cells), "missing or duplicate cells")
        require(results["cells_completed"] == len(expected_cells), "completed-cell count mismatch")
        index_by_key = {key: i for i, key in enumerate(keys)}
        for s in splits:
            fit, cal, test = [set(s[k]) for k in ["fit", "calibration", "test"]]
            require(all(len(s[k]) == len(set(s[k])) for k in ["fit", "calibration", "test"]), "duplicated membership")
            require(fit and cal and test and not (fit & cal or fit & test or cal & test), "partition leakage")
            require(fit | cal | test == set(range(n)), "partition coverage")
            if s["cell"].startswith("repeat"):
                repeat, fold = int(s["cell"].split("_")[1]), int(s["cell"].split("_")[3])
                seed = config["split_seeds"][repeat]
                order = sorted(range(n), key=lambda i: hashlib.sha256(f"{seed}:{keys[i]}".encode()).hexdigest())
                expected_test = set(order[fold::config["n_folds"]])
                require(test == expected_test, "outer fold differs from frozen hash split")
                actual = {index_by_key[r["canonical"]] for r in rows if int(r["repeat"]) == repeat and int(r["fold"]) == fold}
                require(actual == test, "prediction/fold alignment")
                calibration_seed = seed + 731
            else:
                element = s["cell"].removeprefix("holdout_")
                expected_test = {i for i, key in enumerate(keys) if element in {part.split(":")[0] for part in key.split("|")}}
                require(test == expected_test, "element-holdout membership mismatch")
                calibration_seed = 731
            outer = sorted(fit | cal)
            calibration_order = sorted(outer, key=lambda i: hashlib.sha256(f"{calibration_seed}:{keys[i]}".encode()).hexdigest())
            calibration_count = max(1, int(round(len(outer) * config["calibration_fraction"])))
            require(cal == set(calibration_order[:calibration_count]), "calibration membership differs from protocol")
            checkpoint = run / "models" / (s["cell"] + ".joblib")
            require(checkpoint.is_file(), "missing checkpoint")
        models = {"et_mean", "et_median", "et_median_96", "et_fractions_median", "random_forest", "hist_gradient_boosting", "ridge", "train_median"}
        expected_metrics = {(i, m) for i in range(len(config["split_seeds"])) for m in models}
        require({(m["repeat"], m["model"]) for m in results["metrics"]} == expected_metrics and len(results["metrics"]) == len(expected_metrics), "incomplete model metrics")
        for m in results["metrics"]:
            subset = [r for r in rows if int(r["repeat"]) == m["repeat"]]
            scores = metrics(subset, "target", m["model"])
            for k, v in scores.items():
                close(v, m[k], f"{m['model']} {k}")
            recomputed[f"repeat_{m['repeat']}/{m['model']}"] = scores
        repeats = range(len(config["split_seeds"]))
        complete_matrix(results["calibration"], ["repeat", "method", "nominal_coverage"],
                        {(i, method, c) for i in repeats for method in ["mean_normalized", "mean_constant", "median_normalized", "median_constant"] for c in config["interval_coverages"]}, "calibration matrix")
        for c in results["calibration"]:
            subset = [r for r in rows if int(r["repeat"]) == c["repeat"]]
            prefix = f"{c['method']}_{int(c['nominal_coverage']*100)}_"
            for k, v in interval(subset, "target", prefix + "lower", prefix + "upper", c["nominal_coverage"]).items():
                close(v, c[k], "calibration " + k)
        differences = [recomputed[f"repeat_{i}/et_median"]["mae_ev"] - recomputed[f"repeat_{i}/et_mean"]["mae_ev"] for i in range(len(config["split_seeds"]))]
        primary_summary(results["primary"], differences, config)
        decision = "SUPPORTED" if all(d <= -config["minimum_mae_improvement_ev"] for d in differences) else ("NEGATIVE" if all(d >= 0 for d in differences) else "MIXED")
        require(results["disposition"] == decision, "scientific disposition mismatch")
        ledger = table(run / "predictions/ranking.csv")
        require(len(ledger) == len(rows), "incomplete candidate ledger")
        require({(int(r["repeat"]), r["canonical"]) for r in ledger} == {(i, key) for i in repeats for key in keys}, "candidate ledger identity mismatch")
        oof_by_key = {(int(r["repeat"]), r["canonical"]): r for r in rows}
        for r in ledger:
            original = oof_by_key[int(r["repeat"]), r["canonical"]]
            for field, column in [("target", "target"), ("prediction", "et_median"), ("sigma", "sigma"), ("empirical_tree_mass", "tree_mass")]:
                close(r[field], original[column], "candidate " + field)
            close(r["greedy_median"], -abs(float(r["prediction"]) - mean(config["target_window_ev"])), "greedy score")
            mu, sigma = float(r["prediction"]), float(r["sigma"])
            require(sigma >= config["sigma_floor_ev"], "candidate scale below floor")
            lo, hi = config["target_window_ev"]
            probability = .5 * (math.erf((hi-mu)/(sigma*math.sqrt(2))) - math.erf((lo-mu)/(sigma*math.sqrt(2))))
            close(r["gaussian_working"], probability, "Gaussian working score")
        complete_matrix(results["ranking"], ["repeat", "policy", "budget"],
                        {(i, policy, budget) for i in repeats for policy in ["empirical_tree_mass", "gaussian_working", "greedy_median", "random"] for budget in config["ranking_budgets"] if budget <= n}, "ranking matrix")
        prevalence = mean(config["target_window_ev"][0] <= y <= config["target_window_ev"][1] for y in targets.values())
        for summary in results["ranking"]:
            close(summary["precision"], summary["hits"] / summary["budget"], "ranking precision")
            close(summary["enrichment"], summary["precision"] / prevalence, "ranking enrichment")
            if summary["policy"] == "random":
                require(len(summary["replicate_hits"]) == config["random_replicates"], "random repetition count")
                close(mean(summary["replicate_hits"]), summary["hits"], "random mean")
                continue
            subset = [r for r in ledger if int(r["repeat"]) == summary["repeat"]]
            ordered = sorted(subset, key=lambda r: (-float(r[summary["policy"]]), r["canonical"]))
            require([int(r[summary["policy"] + "_rank"]) for r in ordered] == list(range(1, n+1)), "ranking order mismatch")
            hits = sum(config["target_window_ev"][0] <= targets[r["canonical"]] <= config["target_window_ev"][1] for r in ordered[:summary["budget"]])
            close(hits, summary["hits"], "ranking hits")
        holdouts = table(run / "predictions/holdouts.csv")
        expected_holdouts = {(element, model) for element in config["holdout_elements"] for model in ["et_mean", "et_median"]}
        require({(h["element"], h["model"]) for h in results["holdouts"]} == expected_holdouts and len(results["holdouts"]) == len(expected_holdouts), "missing holdout summary")
        for h in results["holdouts"]:
            subset = [r for r in holdouts if r["element"] == h["element"] and r["model"] == h["model"]]
            require(len(subset) == h["n"] and len({r["canonical"] for r in subset}) == len(subset), "holdout coverage")
            expected_keys = {keys[i] for s in splits if s["cell"] == "holdout_" + h["element"] for i in s["test"]}
            require({r["canonical"] for r in subset} == expected_keys, "holdout prediction membership")
            for r in subset:
                close(r["target"], targets[r["canonical"]], "holdout target")
            for k, v in {**metrics(subset, "target", "prediction"), **interval(subset, "target", "lower", "upper", .9)}.items():
                close(v, h[k], "holdout " + k)
        subgroups = table(run / "tables/subgroups.csv")
        for row in subgroups:
            row["repeat"] = int(row["repeat"])
        complete_matrix(subgroups, ["repeat", "model", "subgroup"],
                        {(i, m, g) for i in repeats for m in models for g in ["zero", "positive", "contains_oxygen", "conflicting_duplicates"]}, "subgroup matrix")
        duplicates = {u["canonical"] for u in units if len(u["source_ids"]) > 1}
        for group in subgroups:
            subset = [r for r in rows if int(r["repeat"]) == group["repeat"] and
                      {"zero": float(r["target"]) == 0, "positive": float(r["target"]) > 0,
                       "contains_oxygen": "O" in {part.split(":")[0] for part in r["canonical"].split("|")},
                       "conflicting_duplicates": r["canonical"] in duplicates}[group["subgroup"]]]
            require(len(subset) == int(group["n"]), "subgroup size mismatch")
            for k, v in metrics(subset, "target", group["model"]).items():
                close(v, group[k], "subgroup " + k)
    else:
        rows = table(run / "predictions/oof_predictions.csv")
        require(len(rows) == len(labels) and {r["material_id"] for r in rows} == set(labels), "incomplete or duplicated official predictions")
        official = read(root / contract["extracted_split_path"])
        # The extracted artifact may wrap the task; locate its five-fold map.
        folds = official.get("folds", official.get("matbench_expt_gap", official))
        expected_fold = {identifier: fold for fold, value in folds.items() if isinstance(value, dict) and "test" in value for identifier in value["test"]}
        require(len(expected_fold) == len(labels), "unrecognized official split schema")
        for r in rows:
            formula, target = labels[r["material_id"]]
            close(r["target_gap_ev"], target, "official target")
            require(r["composition"] == formula and r["fold"] == expected_fold[r["material_id"]], "official row/fold identity")
        if results["study_version"] == "development-v1":
            stored = results["official_cv"]["metrics"]
            columns = {"proposed": "proposed_prediction_ev", "ridge": "ridge_prediction_ev", "trivial": "trivial_prediction_ev"}
            cal90 = next(c for c in results["official_cv"]["calibration"] if c["nominal_coverage"] == .9)
            for k, v in interval(rows, "target_gap_ev", "interval_90_lower_ev", "interval_90_upper_ev", .9).items():
                if k in cal90:
                    close(v, cal90[k], "historical 90% intervals")
            ledger = table(run / "predictions/candidate_ledger.csv")
            require(len(ledger) == 4601 and len({r["canonical_composition"] for r in ledger}) == 4601, "historical candidate coverage")
            for r in ledger:
                close(r["evaluation_mean_gap_ev"], mean(labels[i][1] for i in json.loads(r["source_ids"])), "historical candidate label")
            ranking = table(run / "tables/ranking.csv")
            for summary in ranking:
                policy = summary["policy"]
                if policy == "random":
                    continue  # Historical artifact omits per-replicate orders.
                if policy == "proposed_calibrated_probability":
                    score = lambda r: -float(r["target_window_probability"])
                elif policy == "greedy_predicted_property":
                    score = lambda r: abs(float(r["predicted_gap_ev"]) - 1.35)
                else:
                    score = lambda r: -float(r["calibrated_sigma_ev"])
                ordered = sorted(ledger, key=lambda r: (score(r), r["canonical_composition"]))
                hits = sum(1.1 <= float(r["evaluation_mean_gap_ev"]) <= 1.6 for r in ordered[:int(summary["budget"])] )
                close(hits, summary["hits"], "historical ranking hits")
        else:
            stored = results["metrics"]
            columns = {"quantile_point": "quantile_point_prediction_ev", "ridge": "ridge_prediction_ev"}
            for c in results["calibration"]:
                prefix = f"interval_{int(c['nominal_coverage'] * 100)}_"
                for k, v in interval(rows, "target_gap_ev", prefix + "lower_ev", prefix + "upper_ev", c["nominal_coverage"]).items():
                    close(v, c[k], "quantile calibration " + k)
        for model, column in columns.items():
            recomputed[model] = metrics(rows, "target_gap_ev", column)
            for k, v in recomputed[model].items():
                close(v, stored[model][k], model + " " + k)
    return {"run_id": run_id, "status": "NUMERICALLY_VERIFIED", "manifest_files": count,
            "results_sha256": digest(run / "results.json"), "recomputed": recomputed,
            "scope": "artifact/row/metric verification; not independent training replication or scientific certification"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = verify(args.root.resolve(), args.run_id)
    except Exception as exc:
        report = {"run_id": args.run_id, "status": "REJECTED", "error": str(exc)}
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["status"] == "NUMERICALLY_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
