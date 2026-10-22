import gzip
import hashlib
import json
from pathlib import Path

import pytest

from ml4materials.data import (
    DataContractError,
    MaterialRecord,
    duplicate_audit,
    load_data_contract,
    load_official_splits,
    load_records,
    split_leakage_audit,
    validate_group_isolation,
)


@pytest.mark.integration
def test_pinned_dataset_and_official_splits() -> None:
    root = Path(__file__).parents[1]
    contract = load_data_contract(root)
    records = load_records(root, contract)
    splits = load_official_splits(root, contract, records)
    assert len(records) == 4604
    assert len({record.canonical_composition for record in records}) == 4601
    assert set(splits) == {f"fold_{index}" for index in range(5)}
    assert sum(len(fold["test"]) for fold in splits.values()) == 4604


@pytest.mark.integration
def test_upstream_duplicate_and_leakage_audit_is_stable() -> None:
    root = Path(__file__).parents[1]
    contract = load_data_contract(root)
    records = load_records(root, contract)
    duplicates = duplicate_audit(records)
    splits = load_official_splits(root, contract, records)
    leakage = split_leakage_audit(splits, records)
    assert len(duplicates) == 3
    assert sorted(round(row["target_span_ev"], 2) for row in duplicates) == [0.58, 0.64, 2.3]
    assert [row["canonical_composition_overlap_count"] for row in leakage] == [1, 1, 1, 1, 0]


def test_custom_split_rejects_canonical_duplicate_leakage() -> None:
    records = [
        MaterialRecord("a", "Fe2O3", 2.0, "O:3|Fe:2"),
        MaterialRecord("b", "Fe4O6", 2.1, "O:3|Fe:2"),
    ]
    with pytest.raises(DataContractError, match="leaks"):
        validate_group_isolation(["a"], ["b"], records)


def test_missing_target_fails_closed(tmp_path: Path) -> None:
    raw = {"index": [0], "columns": ["composition", "gap expt"], "data": [["GaAs", None]]}
    path = tmp_path / "data.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(raw, handle)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = {
        "source_path": "data.json.gz",
        "source_sha256": digest,
        "expected_rows": 1,
    }
    with pytest.raises(DataContractError, match="missing or non-finite"):
        load_records(tmp_path, contract)

