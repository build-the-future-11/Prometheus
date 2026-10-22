import json
from pathlib import Path


def test_compute_and_scope_are_frozen() -> None:
    root = Path(__file__).parents[1]
    config = json.loads((root / "configs/study.json").read_text())
    assert config["extra_trees"]["n_jobs"] == 1
    assert config["max_rss_mib"] == 512
    assert config["require_no_pageout_increase"] is True
    assert config["element_holdouts"] == ["Li", "Zn", "Ga", "Se", "Te"]
    assert config["candidate_target_window_ev"] == [1.1, 1.6]


def test_only_one_property_task_is_declared() -> None:
    root = Path(__file__).parents[1]
    contract = json.loads((root / "configs/data_contract.json").read_text())
    assert contract["dataset_name"] == "matbench_expt_gap"
    assert contract["target"] == "gap expt"
    assert contract["target_unit"] == "eV"
