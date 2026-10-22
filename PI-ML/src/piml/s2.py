"""S2: source-linear regression and independently certified solver comparisons.

S1 modules remain unchanged. See research/S2_SPECIFICATION.md for the contract.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import random
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .experiment import canonical_json, sha256_file, write_json
from .pde import generate_case, operator_matrix
from .statistics import bootstrap_mean_ci

METHODS = (
    "direct", "zero_jacobi", "quadratic_jacobi", "zero_laplace",
    "physics_laplace", "quadratic_laplace", "linear_laplace",
    "structured_laplace", "shuffled_laplace",
)


def features(c: np.ndarray, f: np.ndarray, kind: str) -> np.ndarray:
    if c.shape != (4,) or f.shape != (4,) or not np.isfinite(np.r_[c, f]).all():
        raise ValueError("features require four finite coefficient and source coordinates")
    if kind == "quadratic":
        x = np.r_[c, f]
        return np.r_[1., x, [x[i] * x[j] for i in range(8) for j in range(i, 8)]]
    if kind not in {"linear", "structured"}:
        raise ValueError("unknown feature family")
    phi = np.r_[1., c]
    if kind == "structured":
        phi = np.r_[phi, [c[i] * c[j] for i in range(4) for j in range(i, 4)]]
    return np.outer(phi, f).ravel()


@dataclass
class Predictor:
    weights: np.ndarray
    kind: str

    @classmethod
    def fit(cls, cases: list[dict], kind: str, ridge: float, shuffle_seed=None):
        if not cases or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("positive ridge and nonempty training data required")
        x = np.stack([features(np.array(c["coefficients"]), np.array(c["source"]), kind) for c in cases])
        y = np.stack([np.array(c["truth"]) for c in cases])
        if shuffle_seed is not None:
            order = list(range(len(y)))
            random.Random(shuffle_seed).shuffle(order)
            y = y[order]
        penalty = np.full(x.shape[1], ridge)
        if kind == "quadratic":
            penalty[0] *= 1e-6
        # One augmented least-squares solve for all output coordinates; avoids
        # squared conditioning and repeated factorizations of normal equations.
        a = np.vstack([x, np.diag(np.sqrt(penalty))])
        b = np.vstack([y, np.zeros((x.shape[1], y.shape[1]))])
        weights, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
        if rank != x.shape[1] or not np.isfinite(weights).all():
            raise ArithmeticError("rank-deficient/nonfinite fitted model")
        return cls(weights, kind)

    def predict(self, case: dict) -> np.ndarray:
        return features(np.array(case["coefficients"]), np.array(case["source"]), self.kind) @ self.weights


def system(case: dict):
    lower, diagonal, upper = operator_matrix(tuple(case["diffusivity"]))
    return np.array(lower), np.array(diagonal), np.array(upper), np.array(case["forcing"])[1:-1]


def matvec(lower, diagonal, upper, x):
    y = diagonal * x
    y[1:] += lower * x[:-1]
    y[:-1] += upper * x[1:]
    return y


def factor(lower, diagonal, upper):
    d, u, multipliers = diagonal.copy(), upper.copy(), np.empty(len(lower))
    if not all(np.isfinite(v).all() for v in (lower, diagonal, upper)):
        raise ValueError("nonfinite matrix")
    for i in range(len(lower)):
        if d[i] <= 0:
            raise ArithmeticError("nonpositive SPD pivot")
        multipliers[i] = lower[i] / d[i]
        d[i + 1] -= multipliers[i] * u[i]
    if d[-1] <= 0:
        raise ArithmeticError("nonpositive SPD pivot")
    return multipliers, d, u


def factored_solve(factors, b):
    l, d, u = factors
    y = b.copy()
    for i in range(1, len(y)):
        y[i] -= l[i - 1] * y[i - 1]
    y[-1] /= d[-1]
    for i in range(len(y) - 2, -1, -1):
        y[i] = (y[i] - u[i] * y[i + 1]) / d[i]
    return y


def pcg(lower, diagonal, upper, b, initial, preconditioner, rtol, atol, maxiter):
    n = len(b)
    if (n < 1 or initial.shape != b.shape or not np.isfinite(initial).all()
            or not np.isfinite(b).all() or not math.isfinite(rtol) or rtol <= 0
            or not math.isfinite(atol) or atol < 0 or maxiter < 1):
        raise ValueError("invalid PCG arguments")
    if preconditioner == "laplace":
        fs = factor(-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1))
        pre = lambda r: factored_solve(fs, r)
    elif preconditioner == "jacobi":
        pre = lambda r: r / diagonal
    else:
        raise ValueError("unknown preconditioner")
    apply = lambda x: matvec(lower, diagonal, upper, x)
    x = initial.copy()
    r = b - apply(x)
    rhs_norm = float(np.linalg.norm(b))
    scale = max(rhs_norm, 1e-30)
    threshold = max(atol, rtol * rhs_norm)
    history = [float(np.linalg.norm(r)) / scale]
    applications, restarts, steps = 1, 0, 0
    if np.linalg.norm(r) <= threshold:
        return x, 0, applications, True, history, restarts
    z = pre(r)
    p = z.copy()
    rz = float(r @ z)
    for steps in range(1, maxiter + 1):
        ap = apply(p)
        applications += 1
        curvature = float(p @ ap)
        if not math.isfinite(curvature) or curvature <= 0 or rz <= 0:
            break
        alpha = rz / curvature
        x += alpha * p
        r -= alpha * ap
        history.append(float(np.linalg.norm(r)) / scale)
        if np.linalg.norm(r) <= threshold:
            r = b - apply(x)
            applications += 1
            if np.linalg.norm(r) <= threshold:
                return x, steps, applications, True, history, restarts
            restarts += 1
            z = pre(r)
            p, rz = z.copy(), float(r @ z)
            continue
        z = pre(r)
        next_rz = float(r @ z)
        p = z + (next_rz / rz) * p
        rz = next_rz
    actual = float(np.linalg.norm(b - apply(x)))
    return x, steps, applications + 1, actual <= threshold, history, restarts


def make_case(seed: int, grid: int, regime: str, identity: str):
    scale = .9 if regime == "coefficient_ood" else .45
    case = generate_case(identity, regime, seed, grid, scale, 1., 4)
    forcing = np.array(case.forcing)
    if regime == "forcing_ood":
        rng = random.Random(seed + 7_000_000)
        xs = np.linspace(0, 1, grid)
        for mode in range(5, 9):
            forcing += rng.uniform(-1., 1.) * np.sin(mode * np.pi * xs)
    record = dict(id=identity, seed=seed, grid=grid, regime=regime,
                  coefficients=list(case.coefficients), source=list(case.source_coefficients),
                  diffusivity=list(case.diffusivity), forcing=forcing.tolist())
    l, d, u, b = system(record)
    record["truth"] = factored_solve(factor(l, d, u), b).tolist()
    return record


def execute(case: dict, model: dict[str, Predictor], method: str, rtol: float, atol: float):
    # Complete online timer includes assembly, prediction, preconditioner setup,
    # solution and independent true-residual acceptance. Model loading/training
    # and data generation are separately reported offline costs.
    start = time.perf_counter_ns()
    l, d, u, b = system(case)
    n = len(b)
    x0 = np.zeros(n)
    if method == "direct":
        x = factored_solve(factor(l, d, u), b)
        steps, apps, history, restarts = 0, 0, [], 0
        flag = True
    else:
        prefix, pre = method.rsplit("_", 1)
        if prefix in model:
            x0 = model[prefix].predict(case)
        elif prefix == "physics":
            # Solve constant-coefficient reference problem with mean face a.
            faces = .5 * (np.array(case["diffusivity"][:-1]) + np.array(case["diffusivity"][1:]))
            coeff = float(faces.mean()) * (case["grid"] - 1) ** 2
            x0 = factored_solve(factor(-np.ones(n - 1), 2*np.ones(n), -np.ones(n - 1)), b / coeff)
        elif prefix != "zero":
            raise ValueError("unknown method")
        x, steps, apps, flag, history, restarts = pcg(l, d, u, b, x0, pre, rtol, atol, 4*n)
    residual = float(np.linalg.norm(b - matvec(l, d, u, x)))
    apps += 1
    rhs = float(np.linalg.norm(b))
    converged = bool(flag and residual <= max(atol, rtol * rhs))
    elapsed = time.perf_counter_ns() - start
    truth = np.array(case["truth"])
    return dict(iterations=steps, operator_applications=apps, converged=converged,
                relative_residual=residual / max(rhs, 1e-30),
                relative_error=float(np.linalg.norm(x-truth))/max(float(np.linalg.norm(truth)), 1e-30),
                initial_relative_error=float(np.linalg.norm(x0-truth))/max(float(np.linalg.norm(truth)), 1e-30),
                elapsed_ns=elapsed, solution=x.tolist(), recursive_history=history, restarts=restarts)


def source_manifest(root: Path):
    return {str(p.relative_to(root)): sha256_file(p)
            for folder in ("src", "configs", "tests")
            for p in sorted((root / folder).rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json"}}


def validate(config):
    if config["phase"] not in {"development", "confirmatory", "smoke"}:
        raise ValueError("unknown phase")
    if config["phase"] == "confirmatory" and not config.get("frozen"):
        raise ValueError("confirmatory protocol is not frozen")
    if len(set(config["grids"])) != len(config["grids"]) or min(config["grids"]) < 5:
        raise ValueError("invalid grids")
    if config["train_count"] < 2 or config["test_count"] < 1 or config["repetitions"] < 1:
        raise ValueError("invalid sample counts")
    if set(config["regimes"]) - {"iid", "coefficient_ood", "forcing_ood"}:
        raise ValueError("unknown regime")
    if len(set(config["regimes"])) != len(config["regimes"]):
        raise ValueError("duplicate regime")
    if not 0 < config["ridge"] < 1 or min(config["rtols"]) <= 0 or config["timing_repeats"] < 1:
        raise ValueError("invalid numerical configuration")
    # Each rep has 100000 seeds; grid ranges are separate; regimes use 1000.
    if config["train_count"] >= 1000 or config["test_count"] >= 1000 or len(config["grids"]) > 9:
        raise ValueError("seed namespace capacity exceeded")


def analyse(rows, config):
    summaries, comparisons = [], []
    for grid, rtol, regime in itertools.product(config["grids"], config["rtols"], config["regimes"]):
        selected = [r for r in rows if r["grid"] == grid and r["rtol"] == rtol and r["regime"] == regime]
        for method in METHODS:
            group = [r for r in selected if r["method"] == method]
            summaries.append(dict(grid=grid, rtol=rtol, regime=regime, method=method, n=len(group),
                iterations_mean=float(np.mean([r["iterations"] for r in group])),
                time_ms_mean=float(np.mean([r["elapsed_ns"] for r in group])) / 1e6,
                relative_error_mean=float(np.mean([r["relative_error"] for r in group])),
                residual_max=max(r["relative_residual"] for r in group),
                failures=sum(not r["converged"] for r in group)))
        lookup = {(r["rep"], r["case_id"], r["method"]): r for r in selected}
        for proposed, baseline in [("structured_laplace", "zero_laplace"),
                                   ("structured_laplace", "physics_laplace"),
                                   ("structured_laplace", "quadratic_laplace"),
                                   ("structured_laplace", "shuffled_laplace"),
                                   ("structured_laplace", "direct"),
                                   ("zero_laplace", "zero_jacobi")]:
            blocks, timeblocks = [], []
            for rep in range(config["repetitions"]):
                cases = [r for r in selected if r["rep"] == rep and r["method"] == proposed]
                reductions, timing = [], []
                for r in cases:
                    b = lookup[rep, r["case_id"], baseline]
                    reductions.append((b["iterations"] - r["iterations"])/max(b["iterations"], 1))
                    timing.append((b["elapsed_ns"] - r["elapsed_ns"])/b["elapsed_ns"])
                blocks.append(float(np.mean(reductions)))
                timeblocks.append(float(np.mean(timing)))
            effect, lo, hi = bootstrap_mean_ci(blocks, config["bootstrap_seed"], config["bootstrap_repeats"])
            te, tl, th = bootstrap_mean_ci(timeblocks, config["bootstrap_seed"]+1, config["bootstrap_repeats"])
            comparisons.append(dict(grid=grid, rtol=rtol, regime=regime, proposed=proposed, baseline=baseline,
                iteration_reduction=effect if baseline != "direct" else None,
                ci95=[lo, hi] if baseline != "direct" else None,
                training_block_effects=blocks if baseline != "direct" else None,
                online_time_reduction=te, time_ci95=[tl, th], training_block_time_effects=timeblocks,
                all_converged=all(r["converged"] for r in selected if r["method"] in {proposed,baseline})))
    gate = config["primary"]
    primary = [c for c in comparisons if c["grid"] == gate["grid"] and c["rtol"] == gate["rtol"]
               and c["regime"] == "iid" and c["proposed"] == "structured_laplace"
               and c["baseline"] in {"zero_laplace", "physics_laplace"}]
    passed = len(primary) == 2 and all(c["all_converged"] and c["iteration_reduction"] >= gate["minimum_mean"]
              and c["ci95"][0] >= gate["minimum_lower"] for c in primary)
    return dict(summaries=summaries, comparisons=comparisons,
                primary_passed=passed, disposition=("SUPPORTED" if passed else "NEGATIVE")
                if config["phase"] == "confirmatory" else "DEVELOPMENT_ONLY",
                inference_scope="Three independent training blocks; bootstrap is descriptive and low precision. "
                "No iteration ratio against direct (zero iterations) has scientific meaning; use timing only.")


def run(config, output: Path, root: Path, freeze_path: Path | None = None):
    validate(config)
    source = source_manifest(root)
    config_hash = hashlib.sha256(canonical_json(config).encode()).hexdigest()
    if config["phase"] == "confirmatory":
        if freeze_path is None:
            raise ValueError("confirmatory run requires a source-and-protocol freeze")
        freeze = json.loads(freeze_path.read_text())
        if freeze["source"] != source or freeze["config_sha256"] != config_hash:
            raise ValueError("frozen source or protocol has changed")
    output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    write_json(output / "config.json", config)
    write_json(output / "source_manifest.json", source)
    environment = dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
        cpu_count=os.cpu_count(), kaggle=Path('/kaggle/working').is_dir(),
        thread_environment={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")})
    write_json(output / "environment.json", environment)
    rows, data, models, costs = [], [], {}, []
    try:
        for rep, grid in itertools.product(range(config["repetitions"]), config["grids"]):
            base = config["seed_base"] + rep*100000 + config["grids"].index(grid)*10000
            began = time.perf_counter()
            train = [make_case(base+i, grid, "iid", f"train-{rep}-{grid}-{i}") for i in range(config["train_count"])]
            generation_s = time.perf_counter()-began
            for c in train:
                c.update(split="train", rep=rep)
            data.extend(train)
            fitted = {}
            for name in ("quadratic", "linear", "structured", "shuffled"):
                began = time.perf_counter()
                fitted[name] = Predictor.fit(train, "structured" if name == "shuffled" else name,
                                            config["ridge"], base+999 if name == "shuffled" else None)
                costs.append(dict(rep=rep, grid=grid, model=name, fit_seconds=time.perf_counter()-began,
                                  train_generation_seconds=generation_s, parameter_count=fitted[name].weights.size))
            models[f"{rep}-{grid}"] = {k:dict(kind=v.kind,weights=v.weights.tolist()) for k,v in fitted.items()}
            for regime_i, regime in enumerate(config["regimes"]):
                for i in range(config["test_count"]):
                    case = make_case(base + 1000*(regime_i+1)+i, grid, regime, f"eval-{rep}-{grid}-{regime}-{i}")
                    case.update(split="evaluation", rep=rep)
                    data.append(case)
                    for rtol in config["rtols"]:
                        timings = {m: [] for m in METHODS}
                        records = {}
                        for repeat in range(config["timing_repeats"]):
                            order = list(METHODS)
                            random.Random(case["seed"]+repeat*1_000_000).shuffle(order)
                            for method in order:
                                result = execute(case, fitted, method, rtol, config["atol"])
                                timings[method].append(result["elapsed_ns"])
                                if method in records and (records[method]["iterations"] != result["iterations"]
                                       or records[method]["converged"] != result["converged"]):
                                    raise RuntimeError("nondeterministic solver result")
                                records[method] = result
                        for method, result in records.items():
                            result.update(rep=rep, case_id=case["id"], grid=grid, regime=regime, rtol=rtol, method=method,
                                          elapsed_ns=float(np.median(timings[method])), timing_samples_ns=timings[method])
                            rows.append(result)
                    if time.process_time()-cpu > config["cpu_budget_seconds"]:
                        raise TimeoutError("frozen CPU budget exceeded")
            write_json(output / "progress.json", dict(completed_rep=rep, grid=grid, rows=len(rows)))
        for name, records in [("cases",data),("metrics",rows)]:
            with (output / f"{name}.jsonl").open("w") as f:
                for record in records:
                    f.write(canonical_json(record)+"\n")
        write_json(output / "models.json", models)
        write_json(output / "offline_costs.json", costs)
        analysis = analyse(rows, config)
        write_json(output / "analysis.json", analysis)
        write_json(output / "receipt.json", dict(status="COMPLETE", disposition=analysis["disposition"],
            phase=config["phase"], config_sha256=config_hash, case_count=len(data), metric_rows=len(rows),
            wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)))
        write_json(output / "manifest.json", {p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
        checked = verify(output)
        write_json(output / "verification.json", checked)
        if checked["status"] != "PASS":
            raise RuntimeError("semantic verification failed")
        return analysis
    except BaseException as exc:
        write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc), completed_rows=len(rows),
            wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
        if rows:
            write_json(output / "partial_rows.json", rows)
        raise


def verify(output: Path):
    failures = []
    manifest = json.loads((output / "manifest.json").read_text())
    required = {"config.json","source_manifest.json","environment.json","cases.jsonl","metrics.jsonl", "models.json", "offline_costs.json","analysis.json","receipt.json","progress.json"}
    if set(manifest) != required:
        failures.append("manifest-required-files")
    for name, expected in manifest.items():
        p = output / name
        if Path(name).name != name or not p.is_file() or sha256_file(p) != expected:
            failures.append(f"hash:{name}")
    if failures:
        return dict(status="FAIL", failures=failures)
    config = json.loads((output / "config.json").read_text())
    validate(config)
    receipt = json.loads((output / "receipt.json").read_text())
    if receipt["config_sha256"] != hashlib.sha256(canonical_json(config).encode()).hexdigest():
        failures.append("config-digest")
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    cases = [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]
    ids = {c["id"]:c for c in cases}
    expected = config["repetitions"]*len(config["grids"])*len(config["regimes"])*config["test_count"]*len(config["rtols"])*len(METHODS)
    if len(rows) != expected or receipt["metric_rows"] != expected:
        failures.append("row-count")
    if len(ids) != len(cases) or len({c["seed"] for c in cases}) != len(cases):
        failures.append("duplicate-case-or-seed")
    expected_cases = config["repetitions"]*len(config["grids"])*(config["train_count"]+len(config["regimes"])*config["test_count"])
    if len(cases) != expected_cases or receipt["case_count"] != expected_cases:
        failures.append("case-count")
    saved = json.loads((output / "models.json").read_text())
    models = {}
    for rep, grid in itertools.product(range(config["repetitions"]),config["grids"]):
        key = f"{rep}-{grid}"
        models[key] = {k:Predictor(np.array(v['weights']),v['kind']) for k,v in saved[key].items()}
        train = [c for c in cases if c["rep"]==rep and c["grid"]==grid and c["split"]=="train"]
        base = config["seed_base"]+rep*100000+config["grids"].index(grid)*10000
        for name, model in models[key].items():
            refit = Predictor.fit(train, model.kind, config['ridge'], base+999 if name=='shuffled' else None)
            if not np.allclose(model.weights, refit.weights,rtol=1e-10,atol=1e-12):
                failures.append(f"model-refit:{key}:{name}")
    expected_keys = {(c["id"], t, m) for c in cases if c["split"] == "evaluation" for t in config["rtols"] for m in METHODS}
    keys = [(r["case_id"],r["rtol"],r["method"]) for r in rows]
    if set(keys) != expected_keys or len(set(keys)) != len(keys):
        failures.append("case-condition-matrix")
    for case in cases:
        replay = make_case(case["seed"],case["grid"],case["regime"],case["id"])
        for field in ("coefficients","source","diffusivity","forcing","truth"):
            if not np.allclose(case[field],replay[field],rtol=1e-12,atol=1e-13):
                failures.append(f"case-replay:{case['id']}:{field}")
    for row in rows:
        case = ids[row["case_id"]]
        l,d,u,b = system(case)
        x = np.array(row["solution"])
        residual = float(np.linalg.norm(b-matvec(l,d,u,x)))
        normb = float(np.linalg.norm(b))
        true_flag = residual <= max(config["atol"],row["rtol"]*normb)
        error = float(np.linalg.norm(x-np.array(case["truth"])))/max(float(np.linalg.norm(case["truth"])),1e-30)
        if row["converged"] != bool(true_flag) or not np.isfinite(x).all():
            failures.append(f"acceptance:{case['id']}:{row['method']}")
        if not math.isclose(row["relative_residual"],residual/max(normb,1e-30),rel_tol=1e-10,abs_tol=1e-14):
            failures.append("residual-metric")
        if not math.isclose(row["relative_error"],error,rel_tol=1e-10,abs_tol=1e-14):
            failures.append("error-metric")
        if len(row["timing_samples_ns"]) != config["timing_repeats"] or any(t<=0 for t in row["timing_samples_ns"]):
            failures.append("timing-count")
        if row["elapsed_ns"] != float(np.median(row["timing_samples_ns"])):
            failures.append("timing-median")
        replay = execute(case,models[f"{case['rep']}-{case['grid']}"],row['method'],row['rtol'],config['atol'])
        if replay['iterations'] != row['iterations'] or replay['converged'] != row['converged']:
            failures.append("solver-replay")
        if not np.allclose(replay['solution'],row['solution'],rtol=1e-10,atol=1e-12):
            failures.append("solution-replay")
    observed = json.loads((output / "analysis.json").read_text())
    if canonical_json(analyse(rows,config)) != canonical_json(observed):
        failures.append("analysis-replay")
    return dict(status="PASS" if not failures else "FAIL",failures=failures,metric_rows=len(rows),case_count=len(cases))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("run")
    p.add_argument("config",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--root",type=Path,default=Path('.'));p.add_argument("--freeze",type=Path)
    p = sub.add_parser("verify");p.add_argument("output",type=Path)
    args = parser.parse_args()
    result = verify(args.output) if args.command == "verify" else run(json.loads(args.config.read_text()),args.output,args.root,args.freeze)
    print(json.dumps(result,indent=2))
    if args.command == "verify" and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
