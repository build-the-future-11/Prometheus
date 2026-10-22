# Current research results

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

Run: `kaggle-robust-aggregation-v3-r1`. All 15 grouped outer cells and five element holdouts completed.
Units: 4601 canonical compositions; three predefined split repetitions.
Primary endpoint disposition: **SUPPORTED**.

| Model | Mean MAE over three repetitions (eV) |
|---|---:|
| et_fractions_median | 0.542550 |
| et_mean | 0.473870 |
| et_median | 0.425622 |
| et_median_96 | 0.426861 |
| hist_gradient_boosting | 0.548780 |
| random_forest | 0.519949 |
| ridge | 0.815014 |
| train_median | 0.975857 |

Median-minus-mean MAE per repetition: [-0.04692215800875339, -0.04674146304683222, -0.051081635618316225].
Exploratory paired canonical-unit bootstrap 95% interval:
[-0.053930233616922126, -0.04268203325767561] eV. This conditions on fitted models
and does not include full training uncertainty or establish confirmatory significance.

Grouped targets and splits differ from official v1. Compare mean versus median
within v3, not absolute v1 versus v3 scores as an isolated treatment effect.
All baselines, ablations, probes, ranking policies, calibration variants and
holdout outcomes are retained in `runs/kaggle-robust-aggregation-v3-r1/tables/`.

## Verification and runtime

- Hosted software suite: 58 passed, zero failures/errors.
- Synthetic end-to-end check: seven cells and checkpoint/inference invariance.
- Independent standard-library verifier recomputed all model MAE/RMSE, interval
  metrics and holdout metrics; verified labels, exact group partitions, ranks,
  resources and manifest bytes. Historical v1/v2 numerical checks also pass.
- Runtime: 424.3 s; peak RSS
  372.9 MiB; pageout delta zero.
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

## Verified follow-up: exact reproduction and release 0.1.1

The fresh run `kaggle-robust-aggregation-v3-reproduction-r1` used the exact
original frozen source and settings. All 20 checkpoints, prediction files,
splits, units and diagnostic outputs match byte for byte. Scientific summaries
match exactly after excluding timing fields. Runtime was 369.8 seconds at
372.1 MiB peak RSS with zero pageout increase. This is same-operator computational
reproduction, not independent scientific replication or protected confirmation.

The repaired verifier rejects four rehashed corrupted copies accepted by its
predecessor. Release tests: 72 passed. Independent arithmetic recomputed the
5,000-resample confidence interval and 12 random-ranking summaries exactly.
Report generation now preserves publication/authorship state, and packaging
refuses to overwrite the original freeze. Source distributions exclude prior
release archives. See `artifacts/reproduction-comparison-20261021.json`,
`artifacts/adversarial-audit-20261021.json`, and `FLAW_AUDIT.md` (37 findings).

The positive primary endpoint and adverse RMSE, positive-gap and chemical-shift
results are unchanged. Independent external validation and venue-specific
submission remain open. Ryan Gomez is the paper author; no affiliation is listed.
