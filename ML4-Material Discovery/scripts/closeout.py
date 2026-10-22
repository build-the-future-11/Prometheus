"""Reconcile current project summaries from verified evidence; preserve history."""
from pathlib import Path
from copy import deepcopy
import json
from statistics import mean
import xml.etree.ElementTree as ET

from scripts.verify_evidence import verify


def reconcile_state(previous, result, scores, test_count, failures):
    """Update numerical facts without erasing later release or author decisions."""
    state = deepcopy(previous)
    state.setdefault("phase", "BOUNDED_DEVELOPMENT_STUDIES_EXECUTED_RELEASE_PREPARATION")
    state["terminal_scientific_disposition"] = "MIXED_SCOPE_LIMITED_DEVELOPMENT"
    state.setdefault("successor", {}).update(
        study="robust-aggregation-v3", run="kaggle-robust-aggregation-v3-r1",
        status="COMPLETED_NUMERICALLY_VERIFIED", primary_disposition=result["disposition"], mae_ev=scores,
        notebook="https://www.kaggle.com/code/ryangomez010/ml4-material-discovery-robust-aggregation-v3")
    release = state.setdefault("release", {})
    release.setdefault("tests", {}).update(hosted_passed=test_count, failed=failures)
    state.setdefault("next", ["Verify final paper and release package", "Publish scoped GitHub snapshot and verify remote ref"])
    state.setdefault("outstanding_external_requirements", ["Independently operated reproduction", "Protected external or prospective validation", "Venue-specific submission requirements"])
    return state


def main():
    root = Path(__file__).resolve().parents[1]
    run_id = "kaggle-robust-aggregation-v3-r1"
    verification = verify(root, run_id)
    run = root / "runs" / run_id
    result = json.loads((run / "results.json").read_text())
    receipt = json.loads((run / "receipt.json").read_text())
    scores = {model: mean(m["mae_ev"] for m in result["metrics"] if m["model"] == model)
              for model in {m["model"] for m in result["metrics"]}}
    tests = ET.parse(root / "artifacts/hosted-v3/pytest-v3.xml").getroot()
    test_count = sum(int(s.attrib["tests"]) for s in tests.iter("testsuite"))
    failures = sum(int(s.attrib["failures"]) + int(s.attrib["errors"]) for s in tests.iter("testsuite"))
    if failures:
        raise ValueError("hosted test failures")
    state = json.loads((root / "RESEARCH_STATE.json").read_text())
    state = reconcile_state(state, result, scores, test_count, failures)
    (root / "RESEARCH_STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    status_path = root / "PROJECT_STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update({"schema_version": 2, "canonical_state": "RESEARCH_STATE.json",
              "canonical_status": "EXECUTED_DEVELOPMENT_MIXED_NOT_CONFIRMATORY", "latest_verified_run": run_id,
              "primary_aggregation_disposition": result["disposition"], "independent_replication": False,
              "prospective_validation": False, "release_certified": False, "hosted_tests_passed": test_count,
              "verified_metrics": scores, "numerical_audit": "artifacts/numerical-audit-v3-20261021.json"})
    (root / "PROJECT_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    (root / "artifacts/numerical-audit-v3-20261021.json").write_text(json.dumps(verification, indent=2) + "\n")
    report = f'''# Current research results

Updated 21 October 2026. Canonical state: `RESEARCH_STATE.json`.

**Overall: MIXED, bounded development research.** The historical official study
and two successors have executed. New random splits of inspected labels are not
confirmatory evidence. Independent replication and prospective validation are absent.

## Historical studies, preserved unchanged

- `kaggle-development-v1-r1`: five official folds completed; MAE 0.481126 eV
  versus ridge 0.813118 eV. At budget 50, the historical proposed score selected
  38 target-window hits versus greedy's 15. Nominal 90% coverage was 88.879%;
  selenium-holdout coverage fell to 52.5%. Positive point/ranking development
  evidence and mixed generalization. Official canonical overlap remains disclosed.
- `kaggle-quantile-composition-v2-r1`: completed negative experiment. MAE
  0.975951 eV, zero predictions everywhere. The new synthetic diagnostic
  reproduces a constant-initial-gradient mechanism; it does not replace v2.

## Robust aggregation v3

Run: `{run_id}`. All 15 grouped outer cells and five element holdouts completed.
Units: {result['canonical_units']} canonical compositions; three predefined split repetitions.
Primary endpoint disposition: **{result['disposition']}**.

| Model | Mean MAE over three repetitions (eV) |
|---|---:|
''' + "\n".join(f"| {model} | {score:.6f} |" for model, score in sorted(scores.items())) + f'''

Median-minus-mean MAE per repetition: {result['primary']['median_minus_mean_mae_ev_by_repeat']}.
Exploratory paired canonical-unit bootstrap 95% interval:
{result['primary']['percentile_95_interval_ev']} eV. This conditions on fitted models
and does not include full training uncertainty or establish confirmatory significance.

Grouped targets and splits differ from official v1. Compare mean versus median
within v3, not absolute v1 versus v3 scores as an isolated treatment effect.
All baselines, ablations, probes, ranking policies, calibration variants and
holdout outcomes are retained in `runs/{run_id}/tables/`.

## Verification and runtime

- Hosted software suite: {test_count} passed, zero failures/errors.
- Synthetic end-to-end check: seven cells and checkpoint/inference invariance.
- Independent standard-library verifier recomputed all model MAE/RMSE, interval
  metrics and holdout metrics; verified labels, exact group partitions, ranks,
  resources and manifest bytes. Historical v1/v2 numerical checks also pass.
- Runtime: {receipt['duration_seconds']:.1f} s; peak RSS
  {receipt['resources']['peak_observed_rss_mib']:.1f} MiB; pageout delta zero.
- No independent operator or external dataset has replicated these findings.

## Paper and evidence

`paper/main.tex` is the full numerical manuscript. Its generated macros, tables,
figures and source map live under `paper/generated/`. The archived original
style is used with byte-level provenance in `paper/style-provenance.json`.
The flaw-to-repair checklist is `FLAW_AUDIT.md`; the prior-art boundary is
`LITERATURE_AUDIT.md`. Superseded summaries are retained in
`history/pre-closeout-20261021/`; all failed runs remain in `runs/`.

## Completion boundary

The bounded computational questions have terminal outcomes. This is not a
perfect model, a SOTA claim, an independently replicated result, a certified
production model, or a newly discovered material. External validation,
independent reproduction and venue/author approval remain separate requirements.
'''
    followup_report = root / "VERIFICATION_FOLLOWUP.md"
    if followup_report.exists():
        report += "\n" + followup_report.read_text()
    (root / "RESULTS.md").write_text(report)
    (root / "FINAL_RESEARCH_EXECUTION_REPORT.md").write_text(report)
    (root / "MANUSCRIPT.md").write_text("# Manuscript\n\nThe canonical complete paper is `paper/main.tex`, with generated evidence assets in `paper/generated/` and final PDF `paper/main.pdf`.\n\n" + report)
    author = state["release"].get("paper_author", {}).get("name")
    release_checks = []
    if state["release"].get("paper_verified") and state["release"].get("software_verified"):
        release_checks.append("- [x] Final paper and software release verified.")
    publication = state["release"].get("publication")
    if isinstance(publication, dict) and publication.get("verified_payload_commit"):
        release_checks.append("- [x] Private Atlas GitHub payload published and remote identity verified.")
    release_checks.append(f"- [x] Author confirmed as {author}." if author else "- [ ] Final author metadata.")
    followup = state.get("verification_followup", {})
    if followup.get("reproduction_disposition") == "EXACT_SCIENTIFIC_REPRODUCTION":
        release_checks.append("- [x] Fresh same-operator reproduction: all scientific outputs and 20 checkpoints match exactly.")
    if followup.get("status") == "COMPLETED_VERIFIED" and followup.get("adversarial_audit"):
        release_checks.append("- [x] Adversarial verifier repair and independent resampling verification completed.")
    if state["release"].get("tests", {}).get("release_passed"):
        release_checks.append(f"- [x] Current release suite: {state['release']['tests']['release_passed']} tests passed.")
    (root / "RESEARCH_FEATURE_MEGA_CHECKLIST.md").write_text('''# Current completion checklist

The original feature wishlist is preserved in
`history/pre-closeout-20261021/RESEARCH_FEATURE_MEGA_CHECKLIST.md`.
It describes future research, not a requirement to invent unsupported models.

- [x] Canonical research state, history and fixed budget.
- [x] Precise hypothesis, prior-art boundary, falsifier and full model contract.
- [x] Thirty-seven concrete flaws with repairs, limits and evidence requirements.
- [x] Frozen development protocol distinct from inspected predecessor outcomes.
- [x] Fully implemented model, training, inference, checkpoints and safeguards.
- [x] Group-isolated repetitions, matched-tree controls and stronger baselines.
- [x] Ablations, noise/scale/permutation probes and five chemical holdouts.
- [x] Hosted tests and complete synthetic end-to-end dry run.
- [x] Real Kaggle execution with full raw predictions and retained evidence.
- [x] Independent numerical recomputation and source/manifest verification.
- [x] Complete manuscript source, artifact-generated tables and figures.
- [ ] Independently operated scientific reproduction (not supplied by self-verification).
- [ ] Protected external/prospective validation (not restored by new CV splits).
- [ ] Venue-specific submission approval.

Final PDF/build/GitHub receipts are recorded in `RESEARCH_STATE.json` and `artifacts/`.
''' + "\n".join(release_checks) + "\n")
    print(json.dumps({"scores": scores, "tests_passed": test_count, "primary_disposition": result["disposition"]}, indent=2))


if __name__ == "__main__":
    main()
