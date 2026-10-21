"""Pinned Matbench acquisition, validation, and split construction."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from sklearn.model_selection import KFold

from .chemistry import canonical_formula, element_fractions


DATA_URL = "https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz"
DATA_SHA256 = "783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2"
ROW_COUNT = 4604
OFFICIAL_SPLIT_CANONICAL_SHA256 = "69bc38097c78dd752cdc5506d21b08f32bc3def921b2aac6fc3c8681a1adb566"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(root: Path, source_url: str = DATA_URL) -> dict[str, object]:
    raw = root / "data" / "raw" / "matbench_expt_gap.json.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        temporary = raw.with_suffix(raw.suffix + ".tmp")
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "SPARKING/0.1 research-reproducibility-client"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(raw)
    observed = sha256_file(raw)
    if observed != DATA_SHA256:
        raise ValueError(f"dataset digest mismatch: {observed} != {DATA_SHA256}")
    formulas, targets = load_data(root)
    split_manifest = official_split_manifest(len(formulas))
    canonical = json.dumps(split_manifest, sort_keys=True, separators=(",", ":")).encode()
    split_digest = hashlib.sha256(canonical).hexdigest()
    if split_digest != OFFICIAL_SPLIT_CANONICAL_SHA256:
        raise ValueError(f"official split drift: {split_digest}")
    canonical_count = len({canonical_formula(formula) for formula in formulas})
    receipt = {
        "schema_version": 1,
        "dataset": "matbench_expt_gap",
        "row_count": len(formulas),
        "target_min_ev": float(np.min(targets)),
        "target_max_ev": float(np.max(targets)),
        "source_url": source_url,
        "source_sha256": observed,
        "official_split_canonical_sha256": split_digest,
        "canonical_composition_count": canonical_count,
    }
    destination = root / "data" / "dataset_receipt.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_data(root: Path) -> tuple[list[str], np.ndarray]:
    raw = root / "data" / "raw" / "matbench_expt_gap.json.gz"
    if not raw.exists():
        raise FileNotFoundError("dataset is absent; run 'sparking prepare'")
    if sha256_file(raw) != DATA_SHA256:
        raise ValueError("dataset digest mismatch")
    with gzip.open(raw, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("columns") != ["composition", "gap expt"]:
        raise ValueError("unexpected dataset columns")
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise ValueError("unexpected dataset row count")
    formulas: list[str] = []
    targets: list[float] = []
    for formula, target in rows:
        element_fractions(str(formula))
        value = float(target)
        if not np.isfinite(value) or value < 0:
            raise ValueError("invalid target")
        formulas.append(str(formula))
        targets.append(value)
    return formulas, np.asarray(targets, dtype=float)


def row_id(index: int) -> str:
    return f"mb-expt-gap-{index + 1:04d}"


def official_folds(n_rows: int = ROW_COUNT) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows != ROW_COUNT:
        raise ValueError(f"official task requires exactly {ROW_COUNT} rows")
    splitter = KFold(n_splits=5, shuffle=True, random_state=18012019)
    return [(train, test) for train, test in splitter.split(np.arange(n_rows))]


def official_split_manifest(n_rows: int = ROW_COUNT) -> dict[str, dict[str, list[str]]]:
    return {
        f"fold_{fold}": {
            "train": [row_id(int(index)) for index in train],
            "test": [row_id(int(index)) for index in test],
        }
        for fold, (train, test) in enumerate(official_folds(n_rows))
    }


def element_holdout(formulas: list[str], element: str) -> tuple[np.ndarray, np.ndarray]:
    test = np.asarray([element in element_fractions(formula) for formula in formulas], dtype=bool)
    if int(test.sum()) < 100:
        raise ValueError(f"holdout {element} has only {int(test.sum())} rows")
    return np.flatnonzero(~test), np.flatnonzero(test)

