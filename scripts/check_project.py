#!/usr/bin/env python3
"""Build, test, and compare direct vs sdist-derived wheels for one real project.

No scientific campaigns or protected evaluations are launched. Logs and failure
receipts are always retained. Each CI matrix entry has an isolated environment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import venv
import zipfile

PROJECTS = {
    "AdaptivePredictive-ML": ("test", "adaptive_predictive"),
    "Derived-Learning-ML": ("", "derived_learning"),
    "ML4-Material Discovery": ("dev", "ml4materials"),
    "NEUROGENESIS": ("", "neurogenesis"),
    "PI-ML": ("research", "piml"),
    "SPARKING": ("dev", "sparking"),
}
ROOT = Path(__file__).resolve().parents[1]


def payload(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)): raise ValueError("duplicate wheel member")
        return {n: hashlib.sha256(archive.read(n)).hexdigest() for n in sorted(names) if not n.endswith("/")}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--project", required=True, choices=PROJECTS)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    project = ROOT / args.project
    extras, module = PROJECTS[args.project]
    steps = []; start = time.monotonic()
    receipt = {"schema": "prometheus.project-check.v1", "project": args.project,
               "scope": "package build, unit tests, installed imports, wheel payload comparison; not research execution", "steps": steps}
    env = dict(os.environ, PYTHONHASHSEED="0", TZ="UTC", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def run(label, argv, cwd):
        log = output / (label + ".log"); tick = time.monotonic()
        with log.open("w") as stream:
            try:
                result = subprocess.run([str(x) for x in argv], cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
            except subprocess.TimeoutExpired:
                steps.append({"name": label, "status": "TIMEOUT", "log": log.name}); raise
        steps.append({"name": label, "returncode": result.returncode, "seconds": time.monotonic() - tick, "log": log.name})
        if result.returncode: raise RuntimeError(f"{label} failed; inspect {log.name}")
    try:
        receipt["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        env["SOURCE_DATE_EPOCH"] = subprocess.check_output(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory(prefix="prometheus-check-") as tmp:
            tmp = Path(tmp); venv.EnvBuilder(with_pip=True).create(tmp / "env")
            py = tmp / "env/bin/python"
            run("install-build-tools", [py, "-m", "pip", "install", "build>=1.2,<2", "pytest>=8,<9"], tmp)
            run("build-direct", [py, "-m", "build", "--wheel", "--sdist", "--outdir", tmp / "direct"], project)
            wheels = list((tmp / "direct").glob("*.whl")); sdists = list((tmp / "direct").glob("*.tar.gz"))
            if len(wheels) != 1 or len(sdists) != 1: raise ValueError("expected one wheel and one source distribution")
            with tarfile.open(sdists[0]) as archive:
                members = archive.getmembers()
                if sum(m.size for m in members) > 4 * 1024**3: raise ValueError("sdist exceeds four-GiB extraction budget")
                for member in members:
                    path = Path(member.name)
                    if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                        raise ValueError(f"unsafe source distribution member: {member.name}")
                archive.extractall(tmp / "unpacked", filter="data")
            roots = list((tmp / "unpacked").iterdir())
            if len(roots) != 1 or not roots[0].is_dir(): raise ValueError("invalid sdist root")
            run("build-from-sdist", [py, "-m", "build", "--wheel", "--outdir", tmp / "rebuilt"], roots[0])
            rebuilt = list((tmp / "rebuilt").glob("*.whl"))
            if len(rebuilt) != 1: raise ValueError("expected one rebuilt wheel")
            first, second = payload(wheels[0]), payload(rebuilt[0])
            if first != second:
                receipt["payload_differences"] = [k for k in sorted(first.keys() | second.keys()) if first.get(k) != second.get(k)]
                raise ValueError("direct and sdist-derived wheel payloads differ")
            receipt["wheel_payload"] = first
            install = str(rebuilt[0]) + (f"[{extras}]" if extras else "")
            run("install-wheel", [py, "-m", "pip", "install", install], tmp)
            run("dependency-consistency", [py, "-m", "pip", "check"], tmp)
            run("installed-import", [py, "-c", f"import {module},pathlib; p=pathlib.Path({module}.__file__).resolve(); assert 'site-packages' in p.parts, p; print(p)"], tmp)
            run("unit-tests", [py, "-m", "pytest", "tests", "-q", "-m", "not integration", f"--junitxml={output / 'junit.xml'}"], project)
            run("environment", [py, "-m", "pip", "freeze", "--all"], tmp)
        receipt["status"] = "PASS"
    except Exception as exc:
        receipt.update(status="FAILED_RETAINED", error=f"{type(exc).__name__}: {exc}")
    receipt["elapsed_seconds"] = time.monotonic() - start
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
