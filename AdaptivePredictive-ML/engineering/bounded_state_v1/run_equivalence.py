#!/usr/bin/env python3
"""Execute the frozen engineering matrix and retain every compared prediction.

No Kaggle, network, external datasets, or protected outcomes are used. Execute:
python engineering/bounded_state_v1/run_equivalence.py --output /tmp/bounded-run
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from adaptive_predictive.model import DDFSConfig, DriftDirectedFixedShare
from adaptive_predictive.bounded import BoundedDriftDirectedFixedShare, FloatWindow


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def history_bytes(model):
    """Deep Python bytes of exactly three buffers; exclude all other state."""
    seen = set()
    def size(obj):
        if id(obj) in seen: return 0
        seen.add(id(obj)); n = sys.getsizeof(obj)
        if isinstance(obj, FloatWindow):
            n += size(obj._values)
            n += sum(size(getattr(obj, k)) for k in ("capacity", "_start", "_size"))
        elif isinstance(obj, (list, tuple)):
            n += sum(size(x) for x in obj)
        return n
    return sum(size(getattr(model, k)) for k in ("history", "ensemble_errors", "normalized_residuals"))


def state_hash(model):
    h = hashlib.sha256()
    for name in ("weights", "expert_fast_loss", "expert_slow_loss", "cumulative_expert_losses"):
        h.update(getattr(model, name).tobytes())
    h.update(np.asarray([model.fast_loss, model.slow_loss, model.mixability_gap], dtype=np.float64).tobytes())
    for expert in model.experts:
        h.update(expert.theta.tobytes()); h.update(expert.covariance.tobytes())
        h.update(str(expert.reset_count).encode() + b"\0")
    return h.hexdigest()


def run_case(output, mode, season, window, n, seed, start, budget):
    config = replace(DDFSConfig(), interval_window=window)
    models = [DriftDirectedFixedShare(season=season, config=config, mode=mode),
              BoundedDriftDirectedFixedShare(season=season, config=config, mode=mode)]
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    values = 3 * np.sin(2 * np.pi * t / season) + 0.001 * t + rng.normal(0, 0.5, n)
    values[n // 2:] += 5
    name = f"{mode}-s{season}-w{window}-n{n}"
    durations = [0, 0]; compared = 0
    trace_path = output / (name + ".jsonl.gz")
    checkpoints = []
    with trace_path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as trace:
        for step, value in enumerate(values):
            if step % 128 == 0 and time.perf_counter() - start > budget:
                raise TimeoutError("declared engineering wall budget exhausted")
            if models[0].ready():
                predictions = []
                for i, m in enumerate(models):
                    a = time.perf_counter_ns(); predictions.append(m.predict()); durations[i] += time.perf_counter_ns() - a
                if predictions[0] != predictions[1]: raise AssertionError(f"prediction mismatch {name}:{step}")
                if not np.array_equal(models[0]._last_features, models[1]._last_features):
                    raise AssertionError(f"feature mismatch {name}:{step}")
                for i, m in enumerate(models):
                    a = time.perf_counter_ns(); m.update(float(value)); durations[i] += time.perf_counter_ns() - a
                hashes = [state_hash(m) for m in models]
                if hashes[0] != hashes[1]: raise AssertionError(f"state mismatch {name}:{step}")
                row = {"step": step, "target": float(value), "legacy": asdict(predictions[0]),
                       "bounded": asdict(predictions[1]), "legacy_state_sha256": hashes[0], "bounded_state_sha256": hashes[1]}
                trace.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
                compared += 1
            else:
                for m in models: m.observe_initial(float(value))
            b = models[1]
            if len(b.history) > b.history_capacity or len(b.ensemble_errors) > max(8, window) or len(b.normalized_residuals) > window:
                raise AssertionError("capacity violation")
            if step + 1 in {512, 1024, 2048, 4096, 8192, n}:
                checkpoints.append({"observations": step + 1,
                    "legacy_buffer_bytes": history_bytes(models[0]), "bounded_buffer_bytes": history_bytes(b),
                    "legacy_retained_samples": sum(len(getattr(models[0], k)) for k in ("history", "ensemble_errors", "normalized_residuals")),
                    "bounded_retained_samples": sum(len(getattr(b, k)) for k in ("history", "ensemble_errors", "normalized_residuals"))})
    return {"case": name, "mode": mode, "season": season, "interval_window": window,
            "observations": n, "predictions_compared": compared, "exact_prediction_and_state_match": True,
            "input_array_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "legacy_predict_update_seconds": durations[0] / 1e9,
            "bounded_predict_update_seconds": durations[1] / 1e9,
            "retained_sample_capacity": models[1].retained_sample_capacity,
            "memory_checkpoints": checkpoints, "raw_trace": trace_path.name, "raw_trace_sha256": sha(trace_path)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--season", type=int, choices=(1, 7, 24, 168), help="Execute one declared shard; aggregate only with all four")
    p.add_argument("--long-only", action="store_true")
    a = p.parse_args()
    if a.season and a.long_only: p.error("choose one shard selector")
    a.output.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    source = {str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT / "src/adaptive_predictive/model.py", ROOT / "src/adaptive_predictive/bounded.py",
        Path(__file__).resolve(), protocol_path)}
    legacy = (ROOT / "src/adaptive_predictive/model.py").read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(legacy)).encode() + b"\0" + legacy).hexdigest()
    if blob != protocol["legacy_model_git_blob"]: raise ValueError("legacy source differs from frozen reference")
    write(a.output / "manifest.json", {"protocol": protocol, "source_sha256": source,
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "host": "github-actions" if os.getenv("GITHUB_ACTIONS") == "true" else "local",
        "github_sha": os.getenv("GITHUB_SHA"), "season_shard": a.season, "long_only": a.long_only})
    rows = []; start = time.perf_counter()
    try:
        if not a.long_only:
            for season in ([a.season] if a.season else protocol["seasons"]):
                for window in protocol["interval_windows"]:
                    for mode in protocol["modes"]:
                        row = run_case(a.output, mode, season, window, protocol["observations_per_case"], protocol["seed"], start, protocol["compute_budget"]["cpu_wall_seconds"])
                        rows.append(row); print(row["case"], "EXACT", flush=True)
        if a.long_only or a.season is None:
            rows.append(run_case(a.output, protocol["long_stream_mode"], protocol["long_stream_season"], protocol["long_stream_interval_window"], protocol["long_stream_observations"], protocol["seed"], start, protocol["compute_budget"]["cpu_wall_seconds"]))
        write(a.output / "summary.json", {"status": "PASS", "scope": protocol["scope"],
            "complete_matrix": a.season is None and not a.long_only,
            "cases": rows, "total_predictions_compared": sum(r["predictions_compared"] for r in rows),
            "elapsed_seconds": time.perf_counter() - start})
    except BaseException as exc:
        write(a.output / "failure.json", {"status": "FAILED_RETAINED", "error": str(exc), "traceback": traceback.format_exc(), "completed_cases": rows})
        raise
    finally:
        write(a.output / "artifact_manifest.json", {p.name: sha(p) for p in sorted(a.output.iterdir()) if p.is_file() and p.name != "artifact_manifest.json"})

if __name__ == "__main__": main()
