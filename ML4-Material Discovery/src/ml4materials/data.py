"""Pinned data loading, validation, and split integrity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .chemistry import canonical_composition


class DataContractError(ValueError):
    """Raised when source data or a split violates the frozen contract."""


@dataclass(frozen=True)
class MaterialRecord:
    material_id: str
    composition: str
    gap_ev: float
    canonical_composition: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def load_data_contract(root: Path, config_path: str = "configs/data_contract.json") -> dict[str, Any]:
    return load_json(root / config_path)


def _validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise DataContractError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise DataContractError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return actual


def load_records(root: Path, contract: dict[str, Any]) -> list[MaterialRecord]:
    source_path = root / contract["source_path"]
    _validate_hash(source_path, contract["source_sha256"], "dataset artifact")
    with gzip.open(source_path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("columns") != ["composition", "gap expt"]:
        raise DataContractError(f"unexpected dataset columns: {raw.get('columns')!r}")
    rows = raw.get("data")
    if not isinstance(rows, list) or len(rows) != contract["expected_rows"]:
        raise DataContractError(
            f"expected {contract['expected_rows']} rows, observed {len(rows) if isinstance(rows, list) else 'non-list'}"
        )

    records: list[MaterialRecord] = []
    for zero_based, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 2:
            raise DataContractError(f"row {zero_based} is not [composition, target]")
        formula, target = row
        if not isinstance(formula, str) or not formula:
            raise DataContractError(f"row {zero_based} has missing composition")
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
            raise DataContractError(f"row {zero_based} has missing or non-finite target")
        if target < 0:
            raise DataContractError(f"row {zero_based} has negative band gap")
        material_id = f"mb-expt-gap-{zero_based + 1:04d}"
        canonical = canonical_composition(formula)
        records.append(
            MaterialRecord(
                material_id=material_id,
                composition=formula,
                gap_ev=float(target),
                canonical_composition=canonical,
            )
        )
    return records


def load_official_splits(
    root: Path, contract: dict[str, Any], records: list[MaterialRecord]
) -> dict[str, dict[str, list[str]]]:
    split_path = root / contract["split_source_path"]
    _validate_hash(split_path, contract["split_source_sha256"], "official split artifact")
    extracted_path = root / contract.get("extracted_split_path", "")
    if extracted_path.is_file() and contract.get("extracted_split_sha256"):
        _validate_hash(
            extracted_path,
            contract["extracted_split_sha256"],
            "extracted task split manifest",
        )
        splits = load_json(extracted_path)
    else:
        payload = load_json(split_path)
        if payload.get("metadata") != contract["official_split_metadata"]:
            raise DataContractError(
                f"official split metadata drift: {payload.get('metadata')!r}"
            )
        try:
            splits = payload["splits"][contract["dataset_name"]]
        except (KeyError, TypeError) as error:
            raise DataContractError("dataset missing from official split artifact") from error
    validate_splits(splits, records)
    return splits


def validate_splits(
    splits: dict[str, dict[str, list[str]]], records: list[MaterialRecord]
) -> None:
    all_ids = {record.material_id for record in records}
    composition_by_id = {
        record.material_id: record.canonical_composition for record in records
    }
    observed_test: list[str] = []
    if len(splits) != 5:
        raise DataContractError(f"expected 5 official folds, observed {len(splits)}")
    for fold_name, fold in sorted(splits.items()):
        train = fold.get("train")
        test = fold.get("test")
        if not isinstance(train, list) or not isinstance(test, list):
            raise DataContractError(f"{fold_name} train/test must be lists")
        train_ids, test_ids = set(train), set(test)
        if len(train_ids) != len(train) or len(test_ids) != len(test):
            raise DataContractError(f"{fold_name} contains repeated identifiers")
        if train_ids & test_ids:
            raise DataContractError(f"{fold_name} has identifier leakage")
        if train_ids | test_ids != all_ids:
            raise DataContractError(f"{fold_name} does not partition all records")
        observed_test.extend(test)
    if len(observed_test) != len(all_ids) or set(observed_test) != all_ids:
        raise DataContractError("official test folds do not cover every record exactly once")


def duplicate_audit(records: list[MaterialRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[MaterialRecord]] = {}
    for record in records:
        grouped.setdefault(record.canonical_composition, []).append(record)
    rows: list[dict[str, Any]] = []
    for canonical, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        targets = [record.gap_ev for record in group]
        rows.append(
            {
                "canonical_composition": canonical,
                "source_ids": [record.material_id for record in group],
                "source_formulas": [record.composition for record in group],
                "source_targets_ev": targets,
                "target_span_ev": max(targets) - min(targets),
            }
        )
    return rows


def split_leakage_audit(
    splits: dict[str, dict[str, list[str]]], records: list[MaterialRecord]
) -> list[dict[str, Any]]:
    composition_by_id = {
        record.material_id: record.canonical_composition for record in records
    }
    rows: list[dict[str, Any]] = []
    for fold_name, fold in sorted(splits.items()):
        train_groups = {composition_by_id[item] for item in fold["train"]}
        test_groups = {composition_by_id[item] for item in fold["test"]}
        overlap = sorted(train_groups & test_groups)
        rows.append(
            {
                "fold": fold_name,
                "canonical_composition_overlap_count": len(overlap),
                "overlapping_canonical_compositions": overlap,
            }
        )
    return rows


def validate_group_isolation(
    train_ids: list[str], test_ids: list[str], records: list[MaterialRecord]
) -> None:
    composition_by_id = {
        record.material_id: record.canonical_composition for record in records
    }
    train_groups = {composition_by_id[item] for item in train_ids}
    test_groups = {composition_by_id[item] for item in test_ids}
    overlap = train_groups & test_groups
    if overlap:
        raise DataContractError(
            f"custom split leaks {len(overlap)} canonical composition group(s)"
        )


def prepare_manifests(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    records = load_records(root, contract)
    split_path = root / contract["split_source_path"]
    _validate_hash(split_path, contract["split_source_sha256"], "official split artifact")
    payload = load_json(split_path)
    if payload.get("metadata") != contract["official_split_metadata"]:
        raise DataContractError(
            f"official split metadata drift: {payload.get('metadata')!r}"
        )
    try:
        splits = payload["splits"][contract["dataset_name"]]
    except (KeyError, TypeError) as error:
        raise DataContractError("dataset missing from official split artifact") from error
    validate_splits(splits, records)
    metadata_path = root / contract["upstream_metadata_path"]
    metadata_hash = _validate_hash(
        metadata_path, contract["upstream_metadata_sha256"], "upstream metadata"
    )
    provenance = {
        "contract_version": contract["contract_version"],
        "dataset_name": contract["dataset_name"],
        "dataset_version": contract["dataset_version"],
        "row_count": len(records),
        "source_sha256": contract["source_sha256"],
        "split_source_sha256": contract["split_source_sha256"],
        "upstream_metadata_sha256": metadata_hash,
        "identifiers_sha256": hashlib.sha256(
            "\n".join(record.material_id for record in records).encode("utf-8")
        ).hexdigest(),
        "canonical_compositions_sha256": hashlib.sha256(
            "\n".join(record.canonical_composition for record in records).encode("utf-8")
        ).hexdigest(),
        "canonical_composition_count": len(
            {record.canonical_composition for record in records}
        ),
    }
    duplicates = duplicate_audit(records)
    leakage = split_leakage_audit(splits, records)
    extracted_path = root / contract.get(
        "extracted_split_path", "data/manifests/matbench_expt_gap_official_splits.json"
    )
    write_json(extracted_path, splits)
    expected_extracted_hash = contract.get("extracted_split_sha256")
    if expected_extracted_hash and sha256_file(extracted_path) != expected_extracted_hash:
        raise DataContractError("generated task split manifest does not match frozen hash")
    write_json(root / "data/manifests/provenance.json", provenance)
    write_json(root / "data/manifests/canonical_duplicate_audit.json", duplicates)
    write_json(root / "data/manifests/official_split_leakage_audit.json", leakage)
    return provenance


def records_as_dicts(records: list[MaterialRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
