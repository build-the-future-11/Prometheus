"""Development-only discriminator for the failed S1 strict-tolerance endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from piml.experiment import _cases
from piml.model import EnsembleOperator, pcg_solve
from piml.pde import norm
from piml.statistics import bootstrap_mean_ci, mean, median


def main() -> None:
    config = json.loads(Path("configs/development_v1.json").read_text())
    if config["phase"] != "development":
        raise RuntimeError("this audit refuses non-development configurations")
    train = _cases(config, "train")
    iid = _cases(config, "iid_test")
    ood = _cases(config, "ood_test")
    model_spec = config["model"]
    model = EnsembleOperator.fit(
        train,
        float(model_spec["ridge"]),
        int(model_spec["ensemble_members"]),
        int(model_spec["bootstrap_seed"]),
    )
    records = []
    for tolerance in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-6, 1e-8, 1e-10):
        for split, cases in (("iid_test", iid), ("ood_test", ood)):
            reductions: list[float] = []
            zero_iterations: list[float] = []
            warm_iterations: list[float] = []
            warm_errors: list[float] = []
            convergence = []
            for case in cases:
                prediction, _ = model.predict(case)
                zero = pcg_solve(case, rtol=tolerance, atol=1e-12, max_iterations=63)
                warm = pcg_solve(case, prediction, rtol=tolerance, atol=1e-12, max_iterations=63)
                reductions.append((zero.iterations - warm.iterations) / max(zero.iterations, 1))
                zero_iterations.append(float(zero.iterations))
                warm_iterations.append(float(warm.iterations))
                error = tuple(value - truth for value, truth in zip(warm.solution, case.solution))
                warm_errors.append(norm(error) / max(norm(case.solution), 1e-30))
                convergence.append(zero.converged and warm.converged)
            effect, low, high = bootstrap_mean_ci(reductions, 5511 + int(-10 * __import__("math").log10(tolerance)), 5000)
            records.append(
                {
                    "tolerance": tolerance,
                    "split": split,
                    "n": len(cases),
                    "zero_iterations_mean": mean(zero_iterations),
                    "warm_iterations_mean": mean(warm_iterations),
                    "relative_iteration_reduction_mean": effect,
                    "relative_iteration_reduction_median": median(reductions),
                    "relative_iteration_reduction_ci95": [low, high],
                    "warm_relative_l2_mean": mean(warm_errors),
                    "all_converged": all(convergence),
                }
            )
    output = {
        "status": "DEVELOPMENT_ONLY",
        "source_run": "runs/development/S1-v1",
        "question": "Does the warm start help at task-relevant residual tolerances after failing at 1e-10?",
        "records": records,
    }
    target = Path("runs/development/S1-tolerance-audit.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

