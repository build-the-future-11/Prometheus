#!/usr/bin/env python3
"""Frozen S3 real-data GPU experiment with retained failures and test isolation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def classify_registered_result(
    *,
    paired_count: int,
    required_count: int,
    mean_improvement: float | None,
    wins: int,
    minimum_improvement: float,
    minimum_wins: int,
) -> str:
    """Apply the frozen S3 terminal taxonomy without hiding missing cells."""
    if paired_count != required_count or mean_improvement is None:
        return "INCONCLUSIVE"
    improvement_met = mean_improvement >= minimum_improvement
    wins_met = wins >= minimum_wins
    if improvement_met and wins_met:
        return "SUPPORTED"
    if improvement_met or wins_met:
        return "MIXED"
    return "NEGATIVE"


def linux_memory() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0])
    pageouts = 0
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, raw = line.split()
        if key == "pswpout":
            pageouts = int(raw)
            break
    return values["MemAvailable"] // 1024, pageouts


def admit_resources(config: dict[str, Any], output: Path) -> dict[str, Any]:
    minimum = int(config["resource_gate"]["minimum_available_mib"])
    maximum_delta = int(config["resource_gate"]["maximum_pageout_delta"])
    available_before, pageouts_before = linux_memory()
    time.sleep(2)
    available_after, pageouts_after = linux_memory()
    receipt = {
        "available_mib_before": available_before,
        "available_mib_after": available_after,
        "pageouts_before": pageouts_before,
        "pageouts_after": pageouts_after,
        "pageout_delta": pageouts_after - pageouts_before,
        "minimum_available_mib": minimum,
        "maximum_pageout_delta": maximum_delta,
    }
    receipt["admitted"] = available_after >= minimum and receipt["pageout_delta"] <= maximum_delta
    write_json(output, receipt)
    if not receipt["admitted"]:
        raise RuntimeError(f"resource admission refused: {receipt}")
    return receipt


def resolve_dataset(filename: str) -> Path:
    candidates = sorted(Path("/kaggle/input").rglob(filename))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one {filename}, found {candidates}")
    return candidates[0]


def load_frame(path: Path, config: dict[str, Any]):
    import pandas as pd

    frame = pd.read_csv(path)
    required = {
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        config["dataset"]["target"], config["dataset"]["categorical_column"],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset columns missing: {sorted(missing)}")
    if frame[config["dataset"]["target"]].isna().any():
        raise ValueError("target contains missing values")
    return frame


@dataclass
class Prepared:
    train_x: Any
    train_y: Any
    development_x: Any
    development_y: Any
    test_x: Any
    test_y: Any
    combined_x: Any
    combined_y: Any
    groups: list[list[int]]
    target_mean: float
    target_scale: float
    provenance: dict[str, Any]


def prepare(frame, seed: int, config: dict[str, Any]) -> Prepared:
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    n_train = int(len(order) * config["split"]["train_fraction"])
    n_dev = int(len(order) * config["split"]["development_fraction"])
    train_i = order[:n_train]
    dev_i = order[n_train:n_train + n_dev]
    test_i = order[n_train + n_dev:]
    if set(train_i) & set(dev_i) or set(train_i) & set(test_i) or set(dev_i) & set(test_i):
        raise AssertionError("split overlap")

    numeric = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    category = config["dataset"]["categorical_column"]
    vocabulary = config["dataset"]["categorical_vocabulary"]
    train = frame.iloc[train_i]
    medians = train[numeric].median()
    means = train[numeric].fillna(medians).mean()
    scales = train[numeric].fillna(medians).std(ddof=0).replace(0.0, 1.0)
    target = config["dataset"]["target"]
    target_mean = float(train[target].mean())
    target_scale = float(train[target].std(ddof=0))
    if not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("invalid target scale")

    def transform(indices):
        subset = frame.iloc[indices]
        x_numeric = ((subset[numeric].fillna(medians) - means) / scales).to_numpy(dtype="float32")
        raw_categories = subset[category].fillna("__MISSING__")
        unknown = sorted(set(raw_categories) - set(vocabulary))
        if unknown:
            raise ValueError(f"undeclared categories: {unknown}")
        x_category = np.column_stack([(raw_categories == value).to_numpy(dtype="float32") for value in vocabulary])
        x = np.concatenate([x_numeric, x_category], axis=1)
        y = ((subset[target].to_numpy(dtype="float32") - target_mean) / target_scale).astype("float32")
        return x, y

    train_x, train_y = transform(train_i)
    dev_x, dev_y = transform(dev_i)
    test_x, test_y = transform(test_i)
    combined_x = np.concatenate([train_x, dev_x], axis=0)
    combined_y = np.concatenate([train_y, dev_y], axis=0)
    groups = [[0, 1], [2, 3, 4], [5, 6], [7], list(range(8, 8 + len(vocabulary)))]
    provenance = {
        "seed": seed,
        "train_indices_sha256": canonical_hash(train_i.tolist()),
        "development_indices_sha256": canonical_hash(dev_i.tolist()),
        "test_indices_sha256": canonical_hash(test_i.tolist()),
        "preprocessing_sha256": canonical_hash({"medians": medians.to_dict(), "means": means.to_dict(), "scales": scales.to_dict(), "target_mean": target_mean, "target_scale": target_scale, "vocabulary": vocabulary}),
        "counts": {"train": len(train_i), "development": len(dev_i), "test": len(test_i)},
        "test_targets_used_for_selection": False,
    }
    return Prepared(train_x, train_y, dev_x, dev_y, test_x, test_y, combined_x, combined_y, groups, target_mean, target_scale, provenance)


def seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def rmse(prediction, target) -> float:
    import numpy as np
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def train_neural(data: Prepared, method: str, parameters: dict[str, Any], config: dict[str, Any], *, final: bool, seed: int) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed_all(seed)
    device = torch.device("cuda")
    source_x = data.combined_x if final else data.train_x
    source_y = data.combined_y if final else data.train_y
    evaluation_x = data.test_x if final else data.development_x
    evaluation_y = data.test_y if final else data.development_y
    width = int(config["model"]["hidden_width"])

    class AdditiveBlockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Linear(len(group), width), nn.GELU(), nn.Linear(width, 1))
                for group in data.groups
            ])
            self.intercept = nn.Parameter(torch.zeros(()))

        def components(self, x):
            return [block(x[:, group]).squeeze(-1) for block, group in zip(self.blocks, data.groups)]

        def forward(self, x):
            return torch.stack(self.components(x), dim=0).sum(dim=0) + self.intercept

    model = AdditiveBlockMLP().to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    batch_size = int(config["model"]["batch_size"])
    passes = int(config["model"]["total_example_passes"])
    dataset = TensorDataset(torch.from_numpy(source_x), torch.from_numpy(source_y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    loss_fn = nn.MSELoss()
    learning_rate = float(parameters["learning_rate"])
    weight_decay = float(config["model"]["weight_decay"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    if method == "joint_block_mlp":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for _ in range(passes):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
    elif method == "residual_block_mlp":
        block_count = len(model.blocks)
        sweeps, remainder = divmod(passes, block_count)
        schedule = list(range(block_count)) * sweeps + list(range(remainder))
        optimizers = [
            torch.optim.AdamW(
                [{"params": block.parameters(), "weight_decay": weight_decay}, {"params": [model.intercept], "weight_decay": 0.0}],
                lr=learning_rate,
            )
            for block in model.blocks
        ]
        for block_index in schedule:
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with torch.no_grad():
                    components = model.components(batch_x)
                    other = torch.stack([value for index, value in enumerate(components) if index != block_index], dim=0).sum(dim=0)
                    local_target = batch_y - other
                optimizers[block_index].zero_grad(set_to_none=True)
                local_prediction = model.blocks[block_index](batch_x[:, data.groups[block_index]]).squeeze(-1) + model.intercept
                loss = loss_fn(local_prediction, local_target)
                loss.backward()
                optimizers[block_index].step()
    else:
        raise ValueError(method)

    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(evaluation_x).to(device)).cpu().numpy()
    elapsed = time.perf_counter() - started
    state_hash = sha256_bytes(b"".join(tensor.detach().cpu().numpy().tobytes() for tensor in model.state_dict().values()))
    return {
        "standardized_rmse": rmse(prediction, evaluation_y),
        "rmse": rmse(prediction * data.target_scale + data.target_mean, evaluation_y * data.target_scale + data.target_mean),
        "parameter_count": count,
        "example_passes": passes,
        "wall_seconds": elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "state_sha256": state_hash,
    }


def train_classical(data: Prepared, method: str, parameters: dict[str, Any], *, final: bool, seed: int) -> dict[str, Any]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    source_x = data.combined_x if final else data.train_x
    source_y = data.combined_y if final else data.train_y
    evaluation_x = data.test_x if final else data.development_x
    evaluation_y = data.test_y if final else data.development_y
    started = time.perf_counter()
    if method == "ridge":
        model = Ridge(alpha=float(parameters["alpha"]), fit_intercept=True, solver="svd")
    elif method == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=float(parameters["learning_rate"]), max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            max_iter=300, l2_regularization=1e-4, early_stopping=False, random_state=seed,
        )
    else:
        raise ValueError(method)
    model.fit(source_x, source_y)
    prediction = model.predict(evaluation_x)
    elapsed = time.perf_counter() - started
    return {
        "standardized_rmse": rmse(prediction, evaluation_y),
        "rmse": rmse(prediction * data.target_scale + data.target_mean, evaluation_y * data.target_scale + data.target_mean),
        "parameter_count": int(source_x.shape[1] + 1) if method == "ridge" else None,
        "example_passes": None,
        "wall_seconds": elapsed,
        "peak_gpu_bytes": 0,
        "state_sha256": canonical_hash({"method": method, "parameters": parameters, "prediction": np.asarray(prediction).tolist()}),
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import sklearn
    import torch

    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "derived-learning.real-data-gpu.v1":
        raise ValueError("unsupported config")
    if not torch.cuda.is_available():
        raise RuntimeError("S3 requires a real CUDA device")
    output_dir.mkdir(parents=True, exist_ok=False)
    admission = admit_resources(config, output_dir / "resource_admission.json")
    dataset_path = resolve_dataset(config["dataset"]["filename"])
    frame = load_frame(dataset_path, config)
    dataset_hash = sha256_file(dataset_path)
    attempts_path = output_dir / "selection_attempts.jsonl"
    results_path = output_dir / "results.jsonl"
    selected: dict[str, Any] = {}
    prepared_by_seed: dict[int, Prepared] = {}

    for seed in config["seeds"]:
        data = prepare(frame, int(seed), config)
        prepared_by_seed[int(seed)] = data
        write_json(output_dir / "splits" / f"seed-{seed}.json", data.provenance)
        selected[str(seed)] = {}
        for method, candidates in config["candidate_grid"].items():
            records = []
            for candidate_index, parameters in enumerate(candidates):
                record = {"seed": seed, "method": method, "parameters": parameters, "status": "RUNNING", "test_metrics_present": False}
                try:
                    values = train_neural(data, method, parameters, config, final=False, seed=int(seed) * 100 + candidate_index) if method.endswith("mlp") else train_classical(data, method, parameters, final=False, seed=int(seed))
                    record.update(status="SUCCEEDED", development=values)
                except Exception as error:
                    record.update(status="FAILED", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                append_jsonl(attempts_path, record)
                records.append(record)
            successful = [record for record in records if record["status"] == "SUCCEEDED"]
            if not successful:
                raise RuntimeError(f"all candidates failed for seed={seed} method={method}")
            winner = min(successful, key=lambda record: (record["development"]["standardized_rmse"], canonical_hash(record["parameters"])))
            selected[str(seed)][method] = winner["parameters"]

    write_json(output_dir / "selection.json", {"uses_test_targets": False, "selected": selected})

    for seed in config["seeds"]:
        data = prepared_by_seed[int(seed)]
        for method, parameters in selected[str(seed)].items():
            record = {"seed": seed, "method": method, "parameters": parameters, "status": "RUNNING", "split": "protected_test"}
            try:
                values = train_neural(data, method, parameters, config, final=True, seed=int(seed) * 1000 + 7) if method.endswith("mlp") else train_classical(data, method, parameters, final=True, seed=int(seed))
                record.update(status="SUCCEEDED", test=values)
            except Exception as error:
                record.update(status="FAILED", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            append_jsonl(results_path, record)
            _, pageouts_now = linux_memory()
            if pageouts_now > admission["pageouts_after"]:
                raise RuntimeError(f"pageout guard tripped after {method}/seed-{seed}: {pageouts_now} > {admission['pageouts_after']}")

    results = [json.loads(line) for line in results_path.read_text().splitlines()]
    paired = []
    for seed in config["seeds"]:
        by_method = {record["method"]: record for record in results if record["seed"] == seed and record["status"] == "SUCCEEDED"}
        if "joint_block_mlp" in by_method and "residual_block_mlp" in by_method:
            joint = by_method["joint_block_mlp"]["test"]["rmse"]
            residual = by_method["residual_block_mlp"]["test"]["rmse"]
            paired.append({"seed": seed, "joint_rmse": joint, "residual_rmse": residual, "relative_improvement": (joint - residual) / joint})
    mean_improvement = statistics.fmean(item["relative_improvement"] for item in paired) if paired else None
    wins = sum(item["relative_improvement"] > 0 for item in paired)
    criteria = config["success_criteria"]
    registered_result = classify_registered_result(
        paired_count=len(paired),
        required_count=len(config["seeds"]),
        mean_improvement=mean_improvement,
        wins=wins,
        minimum_improvement=criteria["minimum_relative_rmse_improvement"],
        minimum_wins=criteria["minimum_paired_wins"],
    )
    method_means = {}
    for method in config["candidate_grid"]:
        scores = [record["test"]["rmse"] for record in results if record["method"] == method and record["status"] == "SUCCEEDED"]
        method_means[method] = statistics.fmean(scores) if scores else None
    summary = {
        "study_id": config["study_id"], "paired": paired, "paired_count": len(paired), "paired_wins": wins,
        "mean_relative_improvement": mean_improvement, "method_mean_rmse": method_means,
        "registered_result": registered_result,
        "broad_superiority_claim_allowed": False, "independent_reproduction": False,
    }
    write_json(output_dir / "summary.json", summary)
    environment = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
    }
    write_json(output_dir / "environment.json", environment)
    outputs = [path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "study_id": config["study_id"], "status": "COMPLETE", "dataset_path": str(dataset_path), "dataset_sha256": dataset_hash,
        "config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(config_path.parent.parent / config["protocol_path"]),
        "outputs": {str(path.relative_to(output_dir)): sha256_file(path) for path in sorted(outputs)},
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    summary = execute(arguments.config.resolve(), arguments.output_dir.resolve())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
