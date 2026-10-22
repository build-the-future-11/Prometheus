#!/usr/bin/env python3
"""Bounded research verification; never searches for a favorable scientific result.

Creates a new output directory, retains every failed command, leaves frozen model
sources and historical runs unchanged, and labels real-stream re-execution as
post-outcome reproduction because the existing manuscript already contains it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ["AdaptivePredictive-ML", "CAUSAL-ALPHA", "Derived-Learning-ML", "FIN-ML",
            "META-RenormalizedEngramNeuralNetworks", "ML4-Biocomputation",
            "ML4-Material Discovery", "NEUROGENESIS", "PI-ML", "Project-32000", "SPARKING"]
TEST_PROJECTS = ["AdaptivePredictive-ML", "Derived-Learning-ML", "ML4-Material Discovery", "PI-ML", "SPARKING"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def command(args: list[str], cwd: Path, output: Path, label: str, timeout: int = 300) -> dict:
    start = time.monotonic()
    env = dict(os.environ, PYTHONPATH=str(cwd / "src") + os.pathsep + str(cwd),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    log = output / (label + ".log")
    with log.open("x", encoding="utf-8") as f:
        try:
            proc = subprocess.run(args, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            code = proc.returncode
            status = "PASS" if code == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            code, status = None, "TIMEOUT"
    receipt = {"command": args, "cwd": str(cwd), "status": status, "returncode": code,
               "elapsed_seconds": time.monotonic() - start, "log": log.name, "log_sha256": sha(log)}
    save(output / (label + ".json"), receipt)
    print(json.dumps({"step": label, **receipt}), flush=True)
    return receipt


def require(receipt: dict) -> None:
    if receipt["status"] != "PASS":
        raise RuntimeError("Required step failed; retained log: " + receipt["log"])


def inventory() -> list[dict]:
    entries = subprocess.check_output(["git", "ls-tree", "-z", "HEAD"], cwd=ROOT).split(b"\0")
    tree = {}
    for entry in entries:
        if entry:
            meta, name = entry.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            tree[name.decode()] = {"mode": mode, "kind": kind, "oid": oid}
    result = []
    for name in PROJECTS:
        row = {"project": name, "tree": tree.get(name), "scientific_completion": "NOT_ESTABLISHED_BY_INVENTORY"}
        p = ROOT / name
        if tree.get(name, {}).get("mode") == "160000":
            row.update(source_status="UNRESOLVED_GITLINK", next_action="Resolve exact source remote and pinned commit; do not guess or expose private code.")
        elif not any(p.rglob("*.py")):
            row.update(source_status="NO_PYTHON_SOURCE", next_action="Recover authoritative source; bytecode and names are not an implementation.")
        else:
            row.update(source_status="SOURCE_PRESENT", python_files=sum(1 for _ in p.rglob("*.py")))
        state = p / "RESEARCH_STATE.json"
        if state.is_file():
            data = json.loads(state.read_text())
            row["reported_phase"] = data.get("phase", data.get("status"))
            row["reported_disposition"] = data.get("disposition", data.get("scientific_disposition"))
            row["state_sha256"] = sha(state)
        elif (p / "RESEARCH_STATE.md").is_file():
            row["state_sha256"] = sha(p / "RESEARCH_STATE.md")
        result.append(row)
    return result


def engineering(output: Path) -> list[dict]:
    results = []
    stage = output / "derived-preflight-source"
    source = ROOT / "Derived-Learning-ML"
    shutil.copytree(source, stage, ignore=shutil.ignore_patterns("runs", "releases", ".git", "__pycache__", "*.pyc"))
    require(command(["bash", "scripts/build_s3_source_bundle.sh"], stage, output, "derived-build-fresh-source"))
    # Original repository releases are never replaced. This is a newly hashed
    # engineering package; no protected S3 training or Kaggle upload occurs.
    study = source / "scripts/run_s3_gpu_study.py"
    expected = "0407b6d9f34b3c2997a27e2f69fe289a8cfa42d97b95ba4d76cebcba1cdc2970"
    if sha(study) != expected:
        raise RuntimeError("Frozen S3 study source no longer matches its declared identity")
    save(output / "derived-preflight-provenance.json", {
        "study_source_unchanged": True, "study_sha256": expected,
        "bundle_sha256": sha(stage / "releases/s3-r0/dlml-s3-source.bundle"),
        "remote_source_update_required": True, "kaggle_uploaded": False,
        "protected_training_executed": False,
        "change": "Register dynamically loaded test modules in sys.modules before exec_module."
    })
    for name in TEST_PROJECTS:
        cwd = stage if name == "Derived-Learning-ML" else ROOT / name
        slug = name.replace(" ", "_")
        xml = output / (slug + "-junit.xml")
        item = command([sys.executable, "-m", "pytest", "tests", "-q", "--disable-warnings", "--junitxml=" + str(xml)], cwd, output, slug + "-tests", 300)
        item["project"] = name
        if xml.is_file():
            suites = ET.parse(xml).getroot()
            leaves = [suites] if suites.tag == "testsuite" else list(suites.iter("testsuite"))
            item["counts"] = {k: sum(int(s.get(k, "0")) for s in leaves) for k in ("tests", "errors", "failures", "skipped")}
        results.append(item)
    v4 = ROOT / "NEUROGENESIS/successors/memory_v4"
    item = command([sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"], v4, output, "neurogenesis-v4-tests")
    item["project"] = "NEUROGENESIS/memory_v4"
    results.append(item)
    save(output / "engineering-results.json", results)
    return results


def memory_reproduction(output: Path, tests: list[dict]) -> dict:
    require(next(r for r in tests if r["project"] == "NEUROGENESIS/memory_v4"))
    v4 = ROOT / "NEUROGENESIS/successors/memory_v4"
    frozen = json.loads((v4 / "runs/freeze.json").read_text())
    for relative, expected in frozen["source"].items():
        if sha(v4 / relative) != expected:
            raise RuntimeError("Frozen model/source drift: " + relative)
    out = output / "memory-v4-reproduction"
    out.mkdir()
    save(out / "REPLICATION_CONTEXT.json", {
        "classification": "POST_OUTCOME_COMPUTATIONAL_REPRODUCTION",
        "reason": "Existing paper/main.tex already reports both Air and Beijing figures although STATUS.json says real pending.",
        "new_blinded_confirmation": False, "independent_investigator_replication": False,
        "backend": "github_actions" if os.getenv("GITHUB_ACTIONS") else "local_cpu",
        "kaggle_execution": False, "model_or_protocol_retuned": False,
        "historical_phase_labels": "Runner retains original protocol labels; they do not make this repetition blinded."
    })
    lock = v4 / "runs/freeze.json"
    for dataset in ("air", "beijing"):
        target = out / dataset
        require(command([sys.executable, "transfer.py", "run", "--dataset", dataset, "--lock", str(lock), "--output", str(target)], v4, output, "memory-" + dataset + "-run", 1800))
        require(command([sys.executable, "transfer.py", "verify", "--input", str(target), "--lock", str(lock), "--output", str(target / "verification-exact.json"), "--replay"], v4, output, "memory-" + dataset + "-exact-replay", 1800))
        require(command([sys.executable, "audit.py", str(target), str(target / "audit.json")], v4, output, "memory-" + dataset + "-arithmetic-audit"))
    confirmation = v4 / "runs/confirmation-02"
    require(command([sys.executable, "experiment.py", "verify", "--input", str(confirmation), "--output", str(out / "synthetic-artifact-reverification.json")], v4, output, "memory-synthetic-artifact-reverification", 600))
    require(command([sys.executable, "audit.py", str(confirmation), str(out / "synthetic-arithmetic-reverification.json")], v4, output, "memory-synthetic-arithmetic-audit", 600))
    require(command([sys.executable, "experiment.py", "analyze", "--input", str(confirmation), "--lock", str(lock), "--output", str(out / "analysis.json")], v4, output, "memory-synthetic-reanalysis", 600))
    assets = out / "paper"
    require(command([sys.executable, "publication/generate.py", "--lock", str(lock), "--analysis", str(out / "analysis.json"), "--confirmation", str(confirmation), "--air", str(out / "air"), "--beijing", str(out / "beijing"), "--output", str(assets)], v4, output, "memory-paper-assets", 600))
    tex = (v4 / "paper/main.tex").read_text()
    tex = tex.replace(r"\usepackage[margin=1in]{geometry}", r"\usepackage{arxiv}")
    tex = tex.replace(r"\date{October 21, 2026}", r"\date{October 22, 2026}")
    tex = tex.replace(r"\begin{document}", r"\renewcommand{\shorttitle}{NEUROGENESIS: persistent-ridge predictor memory}" + "\n" + r"\begin{document}")
    note = (r"\paragraph{Reproduction provenance.} This release reconstructs and verifies existing results; it is not a new blinded confirmation. The repository manuscript already contained both real-stream outcomes while its state file still described them as pending. Both streams were re-executed without retuning, exactly replayed, and audited for arithmetic and label timing. The retained synthetic traces were reverified and reanalyzed, not searched again. Current execution used CPU infrastructure, not Kaggle; same-operator computation is not independent investigator replication. No external submission has been made." + "\n")
    tex = tex.replace(r"\end{abstract}", r"\end{abstract}" + "\n" + note, 1)
    (assets / "main.tex").write_text(tex)
    style = ROOT / "PI-ML/paper/arxiv.sty"
    if b"Refined arXiv-style layout" not in style.read_bytes():
        raise RuntimeError("Recovered author style is not the expected arxiv package")
    shutil.copy2(style, assets / "arxiv.sty")
    for i in (1, 2):
        require(command(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], assets, output, "memory-paper-compile-" + str(i), 120))
    require(command(["pdftoppm", "-r", "80", "-png", "main.pdf", str(out / "page")], assets, output, "memory-paper-render", 120))
    air = json.loads((out / "air/summary.json").read_text())
    beijing = json.loads((out / "beijing/summary.json").read_text())
    for relative, expected in frozen["source"].items():
        if sha(v4 / relative) != expected:
            raise RuntimeError("Frozen source mutated during reproduction: " + relative)
    result = {"disposition": "MIXED_SCOPE_LIMITED", "new_blinded_confirmation": False,
              "air_relative_mse_change": air["relative_mse_change"], "air_update_reduction": air["update_reduction"],
              "beijing_relative_mse_change": beijing["relative_mse_change"], "beijing_update_reduction": beijing["update_reduction"],
              "beijing_candidate_mse": beijing["results"]["candidate"]["test_mse_original_units"],
              "beijing_persistence_mse": beijing["results"]["last_observation"]["test_mse_original_units"],
              "candidate_is_best_beijing_method": False, "kaggle_execution": False,
              "frozen_sources_unchanged": True, "paper_sha256": sha(assets / "main.pdf"),
              "style_sha256": sha(assets / "arxiv.sty"), "visual_qa": "RENDERED_AWAITING_HUMAN_OR_VISION_INSPECTION"}
    save(out / "REPRODUCTION_RESULT.json", result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reproduce-memory", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "inventory.json", inventory())
    save(output / "environment.json", {"python": sys.version, "platform": platform.platform(), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "kaggle_execution": False})
    result = {"portfolio_complete": False, "external_submissions": 0}
    try:
        tests = engineering(output)
        result["engineering_all_passed"] = all(r["status"] == "PASS" for r in tests)
        if args.reproduce_memory:
            result["memory"] = memory_reproduction(output, tests)
        result["status"] = "EXECUTION_FINISHED" if result["engineering_all_passed"] else "PARTIAL_ENGINEERING_FAILURE"
    except Exception as exc:
        import traceback
        result.update(status="BLOCKED_RETAINED", error=str(exc), traceback=traceback.format_exc())
    save(output / "completion.json", result)
    save(output / "manifest.json", {str(f.relative_to(output)): sha(f) for f in sorted(output.rglob("*")) if f.is_file() and f.name != "manifest.json"})
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "EXECUTION_FINISHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
