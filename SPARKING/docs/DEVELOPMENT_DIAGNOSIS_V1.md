# SPARKING: development failure diagnosis

Verified 2026-10-21T12:09:23.844393+05:30. Diagnostic run: `kaggle-diagnostic-v1-r1`.

**Recommendation: hold confirmation. The original direct boosting comparator collapsed to the zero predictor, and the gate is almost inactive without a reliable residual-risk signal. A simple corrected direct boosting comparator substantially outperforms the current residual architecture.**

The original source, frozen v1 protocol, and all successful/failed run receipts are preserved. These are fixed, post-development exploratory diagnostics on Cu/Ag/B; no protected Li/Zn/Ga/Se/Te evaluation or official-fold evaluation was run.

## Measured comparisons

MAE in eV, lower is better. The original rows are reproduced, not replaced. New variants were specified before their diagnostic run; they are not confirmatory results.

| Model / diagnostic | Cu | Ag | B | Macro-MAE |
|---|---:|---:|---:|---:|
| Original direct HGB / median | 0.823718 | 0.965255 | 2.017059 | 1.268677 |
| Original ungated residual | 0.835646 | 0.591118 | 1.431066 | 0.952610 |
| Original best numerical gate (q=.8, p=2) | 0.834748 | 0.591951 | 1.430384 | 0.952361 |
| Original protocol-selected gate (q=.99, p=1) | 0.835633 | 0.591118 | 1.431066 | 0.952606 |
| Direct HGB, squared-error loss | 0.624912 | 0.413116 | 1.243747 | 0.760592 |
| Residual HGB, squared-error loss, ungated | 0.834912 | 0.598679 | 1.321016 | 0.918202 |
| Fixed alternative geometry (q=.8, p=2) | 0.837082 | 0.596068 | 1.406882 | 0.946677 |
| Constant residual weight 0.5 | 0.900389 | 0.634527 | 1.407433 | 0.980783 |
| Ridge / residual weight zero | 0.972632 | 0.695457 | 1.412627 | 1.026905 |

## 1. High severity: the absolute-error direct baseline stalls at zero

The cause is the combination of the target distribution and the hard-coded absolute-error objective in `src/sparking/model.py:28-38`. The three training sets contain 53.46%, 53.45%, and 53.37% exact-zero labels, respectively. Their median is zero, so boosting initializes at zero. In scikit-learn 1.6.1 the absolute-error gradient is +1 only when prediction exceeds target; it is -1 otherwise, including equality. Every initial gradient is therefore -1, with zero variance. This removes the initial gradient contrast needed for useful splits. The residual target does not share this all-nonnegative, zero-median configuration.

Measured evidence: all 180 direct-HGB stages have two leaves per tree, but the final prediction is exactly zero for every training and held-out row in all three scenarios. Thus this is not a table rounding issue, an accidentally reused median metric, or simply too few training iterations. The residual learner has 31 leaves per tree and nonconstant corrections, so it did train.

Changing only the direct learner objective to squared error, retaining the same features and all tree hyperparameters, lowers macro-MAE from 1.268677 to 0.760592 eV. It beats the ungated residual model in each scenario and improves macro-MAE by about 20.2% relative to that model. This is the clearest measured repair candidate.

Source: [version-pinned scikit-learn absolute-error gradient](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/_loss/_loss.pyx.tp#L319-L333). A local synthetic reproducer also exhibits constant predictions under installed scikit-learn 1.9.1, so upgrading the library alone is not an established repair.

**Repair / acceptance:** in a versioned development successor, declare direct-baseline and residual objectives separately; do not blindly change shared `make_hgb`, because that changes both models. Keep the failed absolute-error comparator visible. Require nonconstant learning on a zero-heavy, informative-feature fixture, record training/held-out prediction spread, leaf statistics and median-relative performance, and flag constant fitted models as degenerate comparators. Do not use a protected performance threshold as the validity check.

## 2. High severity: the distance representation barely registers the excluded element

At `src/sparking/model.py:65-70`, support scaling is fitted only to training rows. The excluded element fraction has variance zero, so StandardScaler gives it scale 1. Its fractions remain between 0 and 1, while varying coordinates are standardized. In the observed nearest-neighbor distances, the excluded element contributes only 0.642% (Cu), 0.696% (Ag), and 0.826% (B) of squared distance on average. These are measurements of that coordinate contribution, not a complete decomposition of all chemical novelty.

Training leave-one-out distances have a broad tail, up to about 65-66. The q=.99 thresholds are 9.654, 9.762, and 9.793, whereas maximum query distances are 9.857, 9.099, and 8.624. Consequently the selected gate attenuates just one Cu prediction and no Ag or B predictions: 1 of 824 scenario-level prediction cases. At q=.80 it activates for only 19/390 Cu, 20/196 Ag, and 47/238 B cases.

Source: [StandardScaler zero-variance behavior](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

**Repair / acceptance:** explicitly report unseen-element fraction mass and training element coverage. Evaluate a training-defined composition-aware distance as a successor, with block scales stated and held-out-label-free construction. Require activation/risk diagnostics on declared development shifts. Unknown-element detection alone must not automatically trigger ridge-only prediction: the next finding shows why.

## 3. High severity: distance is not a reliable indicator that residual suppression helps

Spearman correlations between original distance and residual harm (absolute error of ungated minus ridge) are +0.204 for Cu, +0.062 for Ag, and -0.067 for B. These are descriptive, post-development associations, not significance tests. The signal is weak and inconsistent. Even among q=.80 activated cases, the residual helps on average for Ag and B; suppressing it can remove useful information.

Forcing all residual weights to 0.5 worsens macro-MAE from 0.952610 to 0.980783. Removing residuals entirely yields ridge at 1.026905. Thus merely increasing gate power or treating every unknown element as unsafe is not a supported fix.

One fixed alternative geometry was tested: concatenate raw fraction coordinates divided by sqrt(118) with training-standardized global descriptors divided by sqrt(14), then use q=.8/p=2. This normalization gives equal nominal per-coordinate block averaging, not equal empirical block variance. It activates on 33.6%/27.0%/67.6% of Cu/Ag/B cases and improves macro-MAE only to 0.946677, a 0.005933 eV gain. Cu and Ag get worse; B improves. It remains far behind direct squared-error boosting and is not an accepted solution.

**Repair / acceptance:** select any future risk gate using nested, training-only pseudo-element exclusions or cross-fitted residual-error diagnostics. Compare against constant shrinkage, ridge, ungated residual, and the corrected direct baseline with matching budgets. Freeze the construction and evaluation matrix in a successor before another benchmark campaign. A gate should demonstrate useful risk ordering and a stable effect across development scenarios, not just a larger activation rate.

## 4. Medium severity: the frozen tie rule chooses an almost-off gate

All eight v1 candidates span just 0.000313445 eV, below the 0.002 eV tolerance. The documented preference for lower power followed by higher quantile therefore chooses q=.99/p=1, whose improvement is 0.000004417 eV. The implementation follows the protocol correctly. This is an intentionally conservative selection rule, not a coding bug and not permission to replace the winner after seeing results.

**Repair / acceptance:** retain the v1 choice. In a future protocol, specify the practical improvement required to justify adding a gate and report whether it acts on the intended shifts. Do not shrink the tie tolerance just to promote the numerically best v1 setting.

## 5. Medium severity: tests and evidence did not expose baseline collapse

The original tests verified gate behavior and matched-control algebra, but did not require a direct baseline to learn an informative zero-heavy fixture. The development run retained aggregate metrics and residual checkpoints, but not direct-model per-example predictions or tree health. That is why a completed, manifest-valid run could contain an ineffective comparator.

Added two regression fixtures: reproduce the original constant-prediction failure and demonstrate that squared error learns with the same tree budget. All 14 local tests pass. The hosted diagnostic reran the original 12 tests, recorded all prediction arrays and tree statistics, and reproduced every original aggregate metric within 1e-9. The 48 diagnostic model/scenario metric sets were independently recomputed from saved per-example arrays.

An initial test assertion that every stalled tree must have one leaf failed; it was corrected to test constant predictions instead. Both the hosted real-data run and local fixture show that leaf count alone is not a reliable learning-health check.

**Repair / acceptance:** make prediction spread and fitted-model health part of successor receipts; retain per-example predictions and recompute reported metrics. Pin and verify the executed environment: the original Kaggle run used scikit-learn 1.6.1 while local tests use 1.9.1. This diagnostic required the original hosted NumPy/scikit-learn versions before fitting. Exact environment matching is still not independent scientific reproduction.

## What to do before confirmation

1. Preserve v1 as an honest near-null gate result; do not retune or rewrite its receipt.
2. Start a separately versioned development protocol with the corrected direct squared-error baseline as the comparator to beat. Reconsider whether the ridge-plus-residual architecture is justified: even its squared-error residual variant reaches only 0.918202 eV, and Ag worsens relative to the original residual model.
3. Add explicit unseen-element diagnostics and train-derived risk calibration only if pursuing the gate mechanism. The tested geometry change is insufficient evidence for adoption.
4. Include one bounded development comparison with declared objectives, descriptors, controls, selection rule, environment and artifact checks. Do not keep trying variants until one wins.
5. Freeze a confirmatory protocol only after choosing the final version and deciding whether it has enough development justification to test. Confirmation remains unopened.

## Evidence and verification

- Hosted diagnostics: 20.429 seconds of computation, peak RSS 243.035 MiB, zero pageout increase.
- [Private Kaggle diagnostic run](https://www.kaggle.com/code/ryangomez010/sparking-development-diagnostics-v1)
- [Fixed diagnostic protocol](../protocols/diagnostic-v1.json)
- [Diagnostic script](../scripts/diagnose_development_v1.py)
- [Raw diagnostic results](../runs/kaggle-diagnostic-v1-r1/results.json)
- [Verification receipt](../artifacts/diagnostic-verification-v1.json)
- [Regression tests](../tests/test_development_failure_reproductions.py)
- Per-example arrays: `runs/kaggle-diagnostic-v1-r1/{Cu,Ag,B}-predictions.npz`.
- Source archive, saved launcher, diagnostic protocol/script, output archive, and retained manifest verified. All original results and failures remain intact.
