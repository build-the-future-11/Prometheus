#!/usr/bin/env python3
"""Assemble a self-contained evidence-bound v1 development-baseline package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_tex(analysis: dict) -> str:
    summaries = analysis["policy_summaries"]
    sgd = summaries["sgd_lr_0p01"]
    frozen = summaries["no_adaptation"]
    high_rate = summaries["sgd_lr_0p05"]
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{booktabs}",
        r"\usepackage{hyperref}",
        r"\title{A Fail-Closed Benchmark for Delayed-Label Online Adaptation}",
        r"\author{NEUROGENESIS research workspace}",
        r"\date{2026-10-20}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        "This paper reports a verified development-only study of conventional online adaptation controls on a deterministic sequential regression benchmark with delayed labels, distractor shift, conflict, recurrence, and novelty.",
        f"Across five fixed development identities, delayed SGD at learning rate 0.01 obtained mean prequential MSE {sgd['primary_mse_mean']:.4f} (sample SD {sgd['primary_mse_sample_stdev']:.4f}) versus {frozen['primary_mse_mean']:.4f} (SD {frozen['primary_mse_sample_stdev']:.4f}) for frozen prediction.",
        f"The high-rate SGD control degraded MSE to {high_rate['primary_mse_mean']:.4f}.",
        "The project has no approved learned-rule or memory-weight candidate; this paper makes no efficacy claim for either mechanism and reports no protected evaluation.",
        r"\end{abstract}",
        r"\section{Scope and protocol status}",
        r"The scientific mechanism remains \texttt{SCOPE\_UNRESOLVED}. The retained protocol admits conventional development controls only and locks protected evaluation at zero runs. Each stream has 480 predictions, four informative and four distractor features, four regimes of equal length, and a four-step label delay. Predictions are scored before labels can update a method.",
        r"\section{Verified development evidence}",
        r"All 20 requested attempts completed and no attempt was discarded. The source bundle was independently verified before this package was assembled. The analysis artifact-map digest is \texttt{" + analysis["input_artifact_map_sha256"] + r"}.",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Development-only conventional baseline results. Lower error is better.}",
        r"\label{tab:baseline}",
        r"\input{tables/development_baselines.tex}",
        r"\end{table}",
        r"\section{Interpretation and limitations}",
        f"SGD at 0.01 improved overall development MSE on every paired identity, but its mean first-24 switch MSE was {sgd['adaptation_switch_mse_mean']:.4f}, compared with {frozen['adaptation_switch_mse_mean']:.4f} for frozen prediction; its return-regime MSE was {sgd['retention_return_mse_mean']:.4f}, compared with {frozen['retention_return_mse_mean']:.4f}.",
        "This establishes a stability--plasticity tradeoff in a synthetic, small, development-only setting. It does not establish broad generalization, a learned update rule, a memory mechanism, protected-evaluation performance, independent replication, or publication readiness for a candidate mechanism.",
        r"\section{Artifacts and reproduction}",
        r"The vector figures in \texttt{figures/} show 24-step rolling MSE curves and the switch/return tradeoff. Exact source data, hashes, commands, and the verification receipt are in the reproducibility folder.",
        r"\end{document}",
    ]
    return "\n\n".join(lines) + "\n"


def _readme(analysis: dict) -> str:
    return "\n".join((
        "# NEUROGENESIS v1 development-baseline submission package",
        "",
        "This is a hash-bound snapshot of the verified conventional-baseline study. It is not a candidate-mechanism paper and is **not submission-ready**.",
        "",
        f"- Input artifact map SHA-256: `{analysis['input_artifact_map_sha256']}`",
        f"- Requested/completed development attempts: {analysis['run_accounting']['requested']}/{analysis['run_accounting']['completed']}",
        "- Protected evaluation runs: 0",
        "- Candidate mechanism status: `SCOPE_UNRESOLVED`",
        "",
        "The source paper is `paper/paper.tex`; vector figures and generated tables are beside it. The reproducibility folder contains the original protocol, source code, scripts, configs, and retained bundle.",
        "",
    ))


def _gpu_status() -> str:
    return """# GPU job status

No GPU job is prepared or runnable for the retained v1 study. The admitted 20-run matrix is CPU-only, dependency-free, and completed in milliseconds per attempt. A GPU candidate study requires a dated owner decision selecting exactly one mechanism, a separately versioned v2 protocol, frozen development/evaluation manifests, and resource/falsifier gates. Those prerequisites do not exist, so inventing a GPU command would be scientifically invalid.
"""


def _readiness() -> str:
    return """# Submission-readiness decision

## Decision: NOT SUBMISSION READY for a mechanism-efficacy paper

The development baseline harness, retained artifacts, verifier, source-bound analysis, tables, and vector figures are ready for review. Essential empirical claims remain unverified: no candidate mechanism has been selected or implemented, no candidate-specific controls or ablations exist, no candidate development matrix has run, protected evaluation remains at zero runs, no real-stream generalization study exists, and there is no independent reproduction.

The enclosed manuscript is an evidence-bound benchmark/baseline report only. It must not be submitted as evidence that NEUROGENESIS delivers learned-rule or memory-weight adaptation.
"""


def _paper_table(analysis: dict) -> str:
    labels = {
        "no_adaptation": "Frozen",
        "sgd_lr_0p01": "SGD 0.01",
        "sgd_lr_0p05": "SGD 0.05",
        "mechanism_off_adam_lr_0p01": "Adam 0.01",
    }
    newline = r"\\"
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        f"Policy & Overall MSE & Switch MSE & Return MSE & State bytes & Updates {newline}",
        r"\midrule",
    ]
    for policy, summary in sorted(analysis["policy_summaries"].items()):
        display = labels.get(policy, policy.replace("_", r"\_"))
        lines.append(
            f"{display} & {summary['primary_mse_mean']:.4f} $\\pm$ {summary['primary_mse_sample_stdev']:.4f} & "
            f"{summary['adaptation_switch_mse_mean']:.4f} & {summary['retention_return_mse_mean']:.4f} & "
            f"{summary['peak_state_bytes_max']} & {summary['parameter_updates_total']} {newline}"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", "% Generated from retained artifacts."))
    return "\n".join(lines) + "\n"


def assemble(root: Path, analysis_dir: Path, output: Path) -> None:
    analysis_path = analysis_dir / "analysis.json"
    if not analysis_path.is_file():
        raise ValueError(f"missing analysis artifact: {analysis_path}")
    analysis = _load(analysis_path)
    if analysis.get("schema_version") != "neurogenesis.descriptive-analysis.v1":
        raise ValueError("unsupported analysis schema")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite submission package: {output}")
    output.mkdir(parents=True)
    paper = output / "paper"
    reproducibility = output / "reproducibility"
    shutil.copytree(analysis_dir, output / "analysis")
    shutil.copytree(analysis_dir / "figures", paper / "figures")
    shutil.copytree(analysis_dir / "tables", paper / "tables")
    (paper / "tables" / "development_baselines.tex").write_text(_paper_table(analysis), encoding="utf-8")
    paper.mkdir(exist_ok=True)
    (paper / "paper.tex").write_text(_paper_tex(analysis), encoding="utf-8")
    (paper / "references.bib").write_text("% No external literature claims are made in this evidence-bound report.\n", encoding="utf-8")
    reproducibility.mkdir()
    shutil.copytree(root / "configs", reproducibility / "configs")
    shutil.copytree(root / "neurogenesis", reproducibility / "neurogenesis", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(root / "scripts", reproducibility / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(root / "results" / "development" / "shared-baselines-v1-20261019", reproducibility / "results")
    for name in ("README.md", "REPRODUCE.md", "PROJECT.md", "PROTOCOL.md", "RESULTS.md", "SCOPE_HISTORY.md", "pyproject.toml"):
        shutil.copy2(root / name, reproducibility / name)
    (output / "README.md").write_text(_readme(analysis), encoding="utf-8")
    (output / "GPU_JOBS.md").write_text(_gpu_status(), encoding="utf-8")
    (output / "SUBMISSION_READINESS.md").write_text(_readiness(), encoding="utf-8")
    manifest = {"schema_version": "neurogenesis.submission-package.v1", "analysis_artifact_map_sha256": analysis["input_artifact_map_sha256"], "files": {}}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    assemble(args.root.resolve(), args.analysis.resolve(), args.output.resolve())
    print(json.dumps({"status": "CREATED", "output": args.output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
