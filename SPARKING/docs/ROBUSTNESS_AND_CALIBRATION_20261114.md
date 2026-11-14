# SPARKING: domain robustness and finite-sample calibration repair

Date: 14 November 2026. Baseline: Prometheus commit
`22461b231d4c44fc41562bf24b128a67478524b3`.

## Outcome

The original v3 development result remains **DEVELOPMENT_NOT_SUPPORTED**.
The selected frequency-weighted Extra Trees ensemble improves element-macro MAE
by 0.009123345454 eV over its matched unweighted control, below the strict
0.010 eV admission threshold. The eight existing development domains and their
saved raw predictions were reanalyzed. No new materials model was fitted, and
protected Li/Zn/Ga/Se/Te evaluations and official folds were not opened.

This extension delivers a new domain-sensitivity analysis, a correction to the
conformal-radius primitive, and a fresh synthetic calibration study. The
correction is not evidence of improved materials prediction accuracy.

## 1. Domain sensitivity of the existing result

The original saved-run manifest is valid. We independently recomputed all
320 reported metrics, checked unique row identifiers within each domain, and
verified consistent targets wherever the same row occurs in multiple domains.

| Element | Matched-control MAE minus selected-model MAE, eV |
|---|---:|
| O | -0.009350228 |
| S | 0.015375693 |
| In | -0.003135408 |
| Ge | 0.004185771 |
| Sb | 0.004321663 |
| Ba | 0.009399050 |
| Co | -0.000472179 |
| F | 0.052662401 |

Positive values favor frequency weighting. Five of eight elements improve.
Fluorine supplies **72.15% of the net summed improvement**. Removing fluorine
reduces the remaining seven-domain mean improvement to **0.002903480405 eV**.
This concentration matters even though the original fixed-panel row-cluster
bootstrap interval was positive. That interval describes resampling compositions
within the observed panel; it does not establish performance on new elements.

Applying the original candidate tie rule to each seven-domain subset selects
the same alpha=0.50, mean-aggregation candidate in all eight cases. This shows
local selection stability, but it is a retrospective sensitivity calculation.
Because samples and training data overlap, it is not an independent
cross-validation or confirmatory experiment.

### Exact sensitivity to domain weights

Let d_i be the eight paired improvements above, and let u_i=1/8. We calculate

`min q·d subject to q >= 0, sum(q)=1, 0.5*sum(abs(q-u)) <= epsilon`.

The implemented algorithm transfers probability mass from the largest
improvement to the smallest. Four independent linear-program comparisons verify
the optimizer. This is an exact finite-panel calculation, not a population
confidence interval.

| Total-variation budget epsilon | Worst weighted improvement, eV |
|---:|---:|
| 0.000 | 0.009123345 |
| 0.050 | 0.006022714 |
| 0.100 | 0.002922083 |
| 0.125 | 0.001371767 |
| 0.200 | -0.000482677 |
| 0.250 | -0.001718973 |

The smallest total-variation change that makes the weighted improvement
nonpositive is **0.180478900663**. This quantifies sensitivity to how the observed
domains are weighted; it cannot predict the performance of an unobserved domain.

Paired individual forest seeds give macro improvements of 0.013032278353,
0.007595202215, and 0.006747099659 eV. These are three descriptive seed results,
not three independent datasets and not replacements for the frozen ensemble.

## 2. Corrected conformal rank boundary

The old `conformal_radius` implemented
`rank = min(ceil((n+1)*coverage), n)`. Clamping to n is wrong when the requested
rank is n+1. Under exchangeable continuous residuals, a finite interval whose
radius is the calibration maximum has marginal coverage n/(n+1). With two
calibration residuals, a nominal 90% interval therefore covers only 2/3 on
average.

The corrected primitive returns positive infinity for rank n+1. The production
finite-interval reporting path explicitly refuses this case and records a
failure instead of silently emitting an overconfident finite interval or
nonstandard infinite JSON metrics. Ordinary finite-radius cases retain the
same order statistic. Invalid negative residuals, empty metrics, invalid
coverage values, nonfinite interval endpoints, and invalid bootstrap budgets
are now rejected.

For a fixed scoring function with exchangeable continuous calibration and test
scores, the test score's rank among n+1 scores is uniform. A threshold at the
k-th calibration order statistic therefore covers with probability k/(n+1)
when k<=n; the missing (n+1)-th threshold is infinity. This is the standard
finite-sample rank correction, not a new theorem. See Ryan Tibshirani's
[Conformal Prediction lecture notes](https://www.stat.berkeley.edu/~ryantibs/statlearn-s24/lectures/conformal.pdf)
and Angelopoulos and Bates,
[A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).

## 3. Fresh synthetic calibration study

The deterministic study uses 100,000 repetitions per cell, eight calibration
sizes, three nominal coverage levels, and two test distributions: 48 cells.
It performs 768 direct checks of the production primitive against the
vectorized order-statistic reference. The calibration residuals are iid
Uniform(0,1); the test residual is either Uniform(0,1) or Uniform(0,2).
All measured coverages agree with their exact rank probabilities within six
Monte Carlo standard errors.

| Calibration size | Nominal coverage | Old measured coverage, iid | Corrected outcome |
|---:|---:|---:|---|
| 2 | 90% | 66.592% | Infinite radius required; finite report refused |
| 8 | 90% | 88.888% | Infinite radius required; finite report refused |
| 9 | 90% | 89.973% | Same finite radius; exact coverage 90% |
| 20 | 90% | 90.390% | Same finite radius; exact coverage 19/21 |

An infinite radius covers every finite score and is not a useful accuracy
improvement. For n=9, doubling the test residual scale reduces measured finite
coverage to 44.696%, against an exact 45%. The rank repair does **not** make
exchangeability-based coverage valid under domain shift. Materials-domain
calibration remains a separate unresolved scientific requirement.

## Validation and reproduction

33 selected metrics, calibration-boundary, and optimizer tests passed. The
original saved v3 evidence was independently verified before analysis. Historical
run files were not overwritten. The test environment emitted one unrelated
pytest-plugin deprecation warning. The full GPU/material-featurization suite was
not rerun because this extension changes metric boundary behavior and performs
saved-array analysis only.

From the repository root:

```bash
PYTHONPATH=SPARKING/src python -m pytest SPARKING/tests/test_metrics.py SPARKING/tests/test_calibration_boundaries.py SPARKING/tests/test_robustness_reweighting.py -q
PYTHONPATH=SPARKING/src python SPARKING/scripts/extend_development_robustness.py SPARKING/runs/kaggle-development-v3-r1 --protocol SPARKING/protocols/development-v3.json --output SPARKING/artifacts/development-v3-robustness-20261114.json
PYTHONPATH=SPARKING/src python SPARKING/scripts/study_conformal_rank_boundary.py --output SPARKING/artifacts/conformal-boundary-study-20261114.json
```

The JSON outputs contain numerical results, scope limits, and source hashes.
The frozen v3 gate still fails. A new successor would require its own frozen
development protocol, independent data, and demonstrated progress before
protected evaluation could be admitted.
