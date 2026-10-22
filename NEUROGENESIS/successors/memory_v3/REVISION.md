# Version 3: second and final development mechanism

Observation: all 210 v2 development attempts completed. Best candidate MSE 0.80245 versus 0.33732 for RLS(0.95); fewer updates did not preserve accuracy. No protected outcomes were generated or examined.

Hypothesized cause: a relative-only gate accepts an expert that is less bad than active even when both are inaccurate; an NLMS active model also forms weaker snapshots than the strongest RLS baseline.

Change: use forgetting-factor RLS(0.95) as the active updater and require archived EW loss <= a fixed absolute threshold in addition to the relative gate. RLS covariance advances only on actual updates; skipped steps do not secretly perform covariance updates.

Prediction: fewer stale-gate locks and stronger stored predictors, at increased active-state size and arithmetic.

Discriminating experiment: the same full development baseline grid and eight v3 candidate configurations; subsequent frozen comparison to all-update RLS bank and no-bank RLS. Failure or success remains retained. This is version two of the two-version development budget, not permission to continue searching.

Novelty: remains a narrow empirical combination of established techniques. Threshold calibration is synthetic-development-only.
