"""Adversarial checks for provenance and summary-corruption regressions."""
from copy import deepcopy
import pytest
from scripts.closeout import reconcile_state
from scripts.package_hosted import refuse_existing_freeze
from scripts.verify_evidence import primary_summary, complete_matrix, formula_ratios


def test_regeneration_preserves_author_publication_and_followup():
    before = {"phase": "BOUNDED_DEVELOPMENT_COMPLETE_PUBLISHED", "next": ["External verification"],
              "outstanding_external_requirements": ["Venue requirements"],
              "release": {"paper_author": {"name": "Ryan Gomez", "affiliation": None},
                          "publication": {"verified_payload_commit": "abc"},
                          "tests": {"release_passed": 58}},
              "verification_followup": {"status": "IN_PROGRESS"}}
    original = deepcopy(before)
    after = reconcile_state(before, {"disposition": "SUPPORTED"}, {"et_median": .42}, 58, 0)
    assert before == original
    for key in ["phase", "next", "outstanding_external_requirements", "verification_followup"]:
        assert after[key] == before[key]
    assert after["release"]["paper_author"] == before["release"]["paper_author"]
    assert after["release"]["publication"] == before["release"]["publication"]
    assert after["release"]["tests"]["release_passed"] == 58


@pytest.mark.parametrize("path", ["artifacts/freeze-v3.json", "kaggle/staging-v3/robust-source.bundle"])
def test_packager_refuses_existing_history(tmp_path, path):
    original = tmp_path / path
    original.parent.mkdir(parents=True)
    original.write_bytes(b"immutable historical evidence")
    with pytest.raises(FileExistsError):
        refuse_existing_freeze(tmp_path)
    assert original.read_bytes() == b"immutable historical evidence"


@pytest.mark.parametrize("mutation", ["empty", "truncated", "mean", "threshold", "count", "reversed_interval"])
def test_primary_summary_rejects_corrupt_paper_numbers(mutation):
    value = {"median_minus_mean_mae_ev_by_repeat": [-.04, -.05, -.06], "mean_difference_ev": -.05,
             "required_reduction_ev": .01, "resamples": 5000, "percentile_95_interval_ev": [-.07, -.03]}
    if mutation == "empty": value["median_minus_mean_mae_ev_by_repeat"] = []
    if mutation == "truncated": value["median_minus_mean_mae_ev_by_repeat"].pop()
    if mutation == "mean": value["mean_difference_ev"] = -.5
    if mutation == "threshold": value["required_reduction_ev"] = .001
    if mutation == "count": value["resamples"] = 5
    if mutation == "reversed_interval": value["percentile_95_interval_ev"].reverse()
    with pytest.raises(ValueError):
        primary_summary(value, [-.04, -.05, -.06], {"minimum_mae_improvement_ev": .01, "bootstrap_repetitions": 5000})


def test_missing_or_duplicated_matrix_rows_are_rejected():
    expected = {(0, "a"), (1, "a")}
    for rows in [[], [{"repeat": 0, "model": "a"}], [{"repeat": 0, "model": "a"}] * 2]:
        with pytest.raises(ValueError):
            complete_matrix(rows, ["repeat", "model"], expected, "fixture")


def test_independent_source_formula_identity():
    assert formula_ratios("Ca(OH)2") == formula_ratios("Ca2O4H4")
    assert formula_ratios("Fe0.5O0.5") == formula_ratios("OFe")
    assert formula_ratios("FeO") != formula_ratios("Fe2O3")
    for invalid in ["Ca(OH", "Fe0O", "X+", "", "2Fe"]:
        with pytest.raises(ValueError):
            formula_ratios(invalid)


@pytest.mark.parametrize("corruption", [None, "interval", "random_counts"])
def test_resampling_recomputed_instead_of_trusting_summary(corruption):
    from scripts.verify_resampling import audit_resamples
    units = [{"canonical": str(i), "target": 1.2 + .1*i} for i in range(4)]
    rows = [{"repeat": 0, "canonical": u["canonical"], "target": u["target"],
             "et_mean": u["target"] + .3, "et_median": u["target"] + .1} for u in units]
    config = {"split_seeds": [7], "bootstrap_repetitions": 20, "random_replicates": 2, "target_window_ev": [1.1, 1.6]}
    results = {"primary": {"percentile_95_interval_ev": [-.2, -.2], "mean_difference_ev": -.2},
               "ranking": [{"policy": "random", "budget": 2, "hits": 2, "replicate_hits": [2, 2]}]}
    if corruption == "interval": results["primary"]["percentile_95_interval_ev"] = [-.4, -.3]
    if corruption == "random_counts": results["ranking"][0]["replicate_hits"] = [0, 4]
    if corruption:
        with pytest.raises(ValueError):
            audit_resamples(rows, units, config, results)
    else:
        assert audit_resamples(rows, units, config, results)["random_ranking_cells_checked"] == 1
