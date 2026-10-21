# SPARKING development v1 results

Verified on 2026-10-21T10:28:47.225031+05:30. Run: `kaggle-development-v1-r1`.

[Private Kaggle run](https://www.kaggle.com/code/ryangomez010/sparking-development-v1-20261020)

Subsequent [development-only diagnostics](DEVELOPMENT_DIAGNOSIS_V1.md) resolved
the baseline caveat below: direct absolute-error boosting collapsed to zero
predictions. The historical results and interpretation below are retained;
the follow-up report contains measured repair comparisons and current readiness.

## Comparison

MAE in eV; lower is better. Macro-MAE gives Cu, Ag, and B equal weight. These are development selection outcomes, not protected confirmatory results.

| Model / setting | Cu | Ag | B | Macro-MAE |
|---|---:|---:|---:|---:|
| Median | 0.823718 | 0.965255 | 2.017059 | 1.268677 |
| Ridge | 0.972632 | 0.695457 | 1.412627 | 1.026905 |
| Direct HGB | 0.823718 | 0.965255 | 2.017059 | 1.268677 |
| Ungated residual | 0.835646 | 0.591118 | 1.431066 | 0.952610 |
| Gate q=0.8, power=1.0 | 0.835120 | 0.591615 | 1.430715 | 0.952483 |
| Gate q=0.8, power=2.0 | 0.834748 | 0.591951 | 1.430384 | 0.952361 |
| Gate q=0.9, power=1.0 | 0.835410 | 0.591371 | 1.431107 | 0.952629 |
| Gate q=0.9, power=2.0 | 0.835241 | 0.591542 | 1.431137 | 0.952640 |
| Gate q=0.95, power=1.0 | 0.835491 | 0.591349 | 1.431095 | 0.952645 |
| Gate q=0.95, power=2.0 | 0.835367 | 0.591534 | 1.431121 | 0.952674 |
| Gate q=0.99, power=1.0 | 0.835633 | 0.591118 | 1.431066 | 0.952606 |
| Gate q=0.99, power=2.0 | 0.835620 | 0.591118 | 1.431066 | 0.952601 |

## Interpretation

- Best numerical gate: q=0.80, power=2; macro-MAE 0.952360766 eV versus ungated 0.952609934 eV. Improvement: 0.000249168 eV (0.0262%). Cu and B improve slightly; Ag worsens slightly.
- The eight candidates differ by only 0.000313445 eV. All qualify under the frozen 0.002 eV tie tolerance, so lower power followed by higher quantile selects q=0.99, power=1. Its improvement is 0.000004417 eV.
- Gating has no practically meaningful development gain here. The protected 0.01 eV success criterion has not been tested; do not classify the confirmatory hypothesis from these selection data.
- Direct HGB exactly matches the median baseline on every reported error metric for all three holdouts. This is an observed baseline limitation, not an established explanation of model behavior; per-example direct-HGB predictions were not retained by this development pipeline.
- At the selected gate, Ag and B have gate=1 throughout; Cu mean gate is 0.999947. The selected configuration therefore behaves almost exactly like the ungated model.

## Verification and provenance

- 12 local tests and 12 hosted tests passed.
- All 43 source files matched the uploaded/downloaded Kaggle dataset; the saved launcher matched locally.
- Downloaded output archive SHA-256: `e78850dfb06f7c3b6d349a4807fe3c1f306bf86945b96574ca5878c993707b03`.
- Retained run manifest passes with no missing, extra, or changed files. All 36 matrix rows, finite metrics, aggregate means, and frozen selection rule were checked.
- Development execution: 10.838 seconds; peak RSS 237.031 MiB; pageout delta 0. This timing excludes hosted setup, dependency installation, and tests.
- Hosted environment: Python 3.13.15, NumPy 2.1.3, scikit-learn 1.6.1. The prepared runner installed within pyproject version ranges; it did not enforce uv.lock. The full pip freeze is retained.
- Manifest verification and aggregate checking are not independent numerical reproduction. Protected Li/Zn/Ga/Se/Te evaluation and official five-fold evaluation remain unrun.

## Evidence

- [Raw results](../runs/kaggle-development-v1-r1/results.json)
- [Run receipt](../runs/kaggle-development-v1-r1/receipt.json)
- [Verified comparison](../artifacts/development-comparison-v1.json)
- [Source roundtrip verification](../artifacts/kaggle-source-roundtrip-verification.json)
- [Hosted environment packages](../artifacts/kaggle-development-v1-r1-output/sparking-source/artifacts/pip-freeze-kaggle-development-v1-r1.txt)

## Next stage

Review the weak development signal and baseline limitation, then freeze the selected gate with source and development evidence digests before any protected evaluation. Any material redesign must preserve this run and use a versioned successor.
