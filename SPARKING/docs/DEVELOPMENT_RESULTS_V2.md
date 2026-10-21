# SPARKING v2 development results

## Scientific disposition

**DEVELOPMENT_NOT_SUPPORTED.** The frozen v2 admission rule did not pass. The selected risk gate at temperature 0.40 eV produced element-macro MAE 0.779667 eV, compared with 0.779536 eV for the identical ungated five-seed CatBoost ensemble. The preregistered improvement, defined as ungated MAE minus gated MAE, was therefore -0.000130 eV. Admission required an improvement greater than 0.020 eV.

This is a valid negative development result. The protected Li, Zn, Ga, Se, and Te outcomes remain unopened, and no confirmatory v2 run is authorized.

## Frozen identities and execution

| Item | Verified value |
|---|---|
| Study | `sparking-development-v2-risk-gated-gpu` |
| Source commit | `5a56bb5dc97619730f0fbc6544490cdc69f2c109` |
| Protocol SHA-256 | `c4aa306c7259238075db18ced236e99d59101d57321e733e84c458f2b7237c2e` |
| Source archive SHA-256 | `915dfeaba793c5e70f90a21cb6e9aaa24db57783dfb8d195812e3a2385e04582` |
| Kaggle kernel | `ryangomez010/sparking-development-v2-risk-gated-gpu`, version 1, private |
| Run | `kaggle-development-v2-r1` |
| Accelerator | Two Tesla T4 GPUs, 15,360 MiB each; driver 580.178.04 |
| Hosted tests | 18 passed, 0 failed |
| CatBoost fits | 180: 60 for each of Cu, Ag, and B |
| Runtime | 6,845.90 s (114.10 min) |
| Peak host RSS | 914.55 MiB |
| Pageout delta | 0 |
| Output archive SHA-256 | `cf8aecebf1dec60ccfb48445b25a891b75b2feff7fb37cf0e08774fbdeeb420a` |
| Retained run-tree SHA-256 | `43eb835f5fe7eef682502fee15c3e6fd3326587313fd2a93a6a5fe398cf32c5c` |

The source roundtrip found 40 expected files with no missing, extra, or changed files. The downloaded output archive contained 18 safe members. Its 17 run files matched the direct Kaggle download byte-for-byte. The independent local verifier recomputed 156 metric comparisons and found a valid manifest with no missing, extra, or changed files.

## Preregistered admission decision

| Gate condition | Required | Observed | Result |
|---|---:|---:|---|
| Macro-MAE improvement vs CatBoost | >0.020 eV | -0.000130 eV | Fail |
| Improved development elements | at least 2 of 3 | 2 of 3 | Pass |
| Active fraction per element | 0.01 to 0.90 | 0.133 to 0.193 | Pass |
| Nonconstant CatBoost predictions | required | true | Pass |
| All 11 pseudo exclusions per outer element | required | complete | Pass |
| Finite, manifest-valid evidence | required | verified | Pass |

All conditions were conjunctive, so the failed practical-effect condition stops v2. The selection rule first minimized macro MAE and then chose the largest temperature among candidates within 0.005 eV of the minimum. Temperature 0.05 was numerically best at 0.776451 eV, only 0.003086 eV better than CatBoost. Temperature 0.40 was within the tie band and was therefore selected as the least aggressive gate. Even the numerical best did not approach the 0.020 eV admission threshold.

## Development performance

MAE values are in eV. Macro is the unweighted mean across Cu, Ag, and B.

| Model | Cu | Ag | B | Macro |
|---|---:|---:|---:|---:|
| Extra Trees ensemble | 0.7271 | 0.3708 | 1.0881 | **0.7287** |
| CatBoost single seed | 0.6978 | 0.4019 | 1.2164 | 0.7720 |
| Direct squared-error HGB | **0.6460** | 0.4395 | 1.2329 | 0.7728 |
| Risk gate, T=0.05 | 0.7078 | 0.3950 | 1.2266 | 0.7765 |
| CatBoost ensemble | 0.7095 | 0.3991 | 1.2300 | 0.7795 |
| Selected risk gate, T=0.40 | 0.7115 | 0.3981 | 1.2294 | 0.7797 |
| Ridge | 0.8021 | 0.5484 | 1.3216 | 0.8907 |
| Median | 0.8237 | 0.9653 | 2.0171 | 1.2687 |

For the selected gate, per-element improvements over CatBoost were -0.001921 eV for Cu, +0.000967 eV for Ag, and +0.000563 eV for B. The gate changed predictions, but its changes were too small and inconsistent to establish a practically useful benefit. Extra Trees was the strongest evaluated control, improving macro MAE by 0.050862 eV relative to the CatBoost ensemble. That secondary comparison does not rescue the preregistered risk-gating hypothesis.

## Risk-model diagnostics

The pseudo-exclusion harm model fit its training-derived calibration data moderately well: Spearman correlations were 0.666 for Cu, 0.656 for Ag, and 0.707 for B. Descriptive correlations on the actual outer element holdouts fell to 0.309, 0.527, and 0.240, respectively. This transfer gap is consistent with the gate's failure: ranking harm within declared pseudo shifts did not translate into a material outer-holdout MAE improvement.

The pseudo-exclusion evidence also contains extreme ridge failures, especially for Fe exclusions. Those rows were retained under the frozen equal-element weighting rule. They may have shaped the learned harm surface, but excluding or clipping them after seeing the result would change the hypothesis and requires a separately versioned successor.

## Post hoc descriptive uncertainty

This analysis was not preregistered and does not replace the admission rule. A deterministic 20,000-resample bootstrap sampled unique Matbench row identifiers, preserving the dependence created when one composition appears in multiple element holdouts. The selected gate's macro improvement had a percentile interval of [-0.001964, 0.001428] eV. The bootstrap fraction above zero was 0.4596; none of the 20,000 resamples exceeded the 0.020 eV admission threshold.

Per-element paired-row percentile intervals were Cu [-0.006702, 0.001790] eV, Ag [-0.001664, 0.003509] eV, and B [-0.000391, 0.001447] eV. These intervals are descriptive because the three development elements were selected design scenarios rather than a random sample of elements.

## Claim boundary

The result supports only the statement that the frozen v2 risk gate did not justify protected confirmation on the declared Cu/Ag/B development protocol. It does not show that all risk gating is ineffective, that Extra Trees is state of the art, or that any model discovers stable or synthesizable materials. Official Matbench folds and the protected element outcomes were not evaluated in v2, so no v2 claim about IID preservation or Li/Zn/Ga/Se/Te generalization is available.

## Reproduction and evidence

Run the independent verifier and deterministic secondary analysis with the project interpreter:

```bash
.venv/bin/python scripts/verify_successor_v2.py runs/kaggle-development-v2-r1
MPLCONFIGDIR=/tmp/sparking-matplotlib .venv/bin/python scripts/analyze_successor_v2.py
```

Primary evidence:

- `runs/kaggle-development-v2-r1/`: manifest-bound raw predictions, pseudo-calibration arrays, diagnostics, environment, resource receipts, and aggregate results.
- `artifacts/development-v2-verification.json`: independent download, checksum, manifest, and metric verification.
- `artifacts/development-v2-posthoc-analysis.json`: explicitly post hoc paired and cluster-bootstrap summary.
- `artifacts/kaggle-development-v2-r1-output/`: exact hosted archive, checksum, environment freeze, source-verification receipt, summary, and kernel log.
- `protocols/development-v2.json`: frozen decision rule and full model budget.
