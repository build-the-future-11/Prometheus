# Protocol v0.2 — pre-freeze development successor

## Estimand

For each series and each forecast time in a temporally protected segment, let
\(e_{m,t}=|\hat y_{m,t}-y_t|/s\), where \(s\) is the mean absolute seasonal
difference computed only from the training segment. The primary paired estimand is

\[
\Delta = \operatorname{mean}_{d}\left[
  \operatorname{mean}_{t\in d}(e_{\mathrm{DDFS},t}-e_{\mathrm{FS},t})
\right],
\]

where each dataset/series unit receives equal top-level weight. Negative values favor
DDFS. The primary practical threshold is \(\Delta\le -0.01\) and at least one-percent
relative improvement over constant fixed share.

## Chronological model

1. At time \(t\), construct causal lag/phase features from \(y_{<t}\).
2. Each RLS expert predicts a normalized residual around the shared seasonal-naive
   base, using one preregistered forgetting factor.
3. The current ensemble weights combine expert predictions. No part of \(y_t\) is
   available.
4. Emit the point forecast and a heuristic rolling-residual interval.
5. Reveal \(y_t\), compute capped normalized aggregation losses, and apply exponential
   weights.
6. Constant fixed share redistributes a fixed mass uniformly. RDDFS uses the outcome
   revealed at step \(t\) only to determine weights for \(t+1\). Its target is the
   normalized geometric mean of a short-memory prior and a causal recent-versus-slow
   expert-loss score; the drift statistic controls how far this target departs from
   uniform redistribution.
7. Update every RLS expert on \(y_t\); append immutable per-step diagnostics.

## Stages

- Engineering: synthetic fixtures and tests. Never outcome evidence.
- Development: initial real-data temporal segment; maximum three model revisions and
  twelve registered configurations.
- Confirmation: final protected temporal segment after a digest-bound freeze receipt.
- Replication: second domain or clean-host rerun without changing the frozen claim.

## Baselines and controls

1. Seasonal naive.
2. Uniform average of the exact expert bank.
3. Hedge on the exact expert bank without share.
4. Constant uniform fixed share on the exact expert bank (primary comparator).
5. DDFS (proposed).

Static and task-specific classical baselines may be added before freeze, but cannot
replace the component-matched primary comparator.

## Statistics

- Report raw MAE, MASE, RMSE, and bias by series.
- Primary uncertainty: paired moving-block bootstrap within each series, then equal
  aggregation across series; fixed RNG seed and block length bound to the freeze.
- Report two-sided 95% percentile interval, probability of improvement, and the
  predefined practical threshold.
- Secondary interval coverage/width is descriptive only.
- No timestamp is dropped because a method performs poorly; non-finite output makes
  the affected run invalid.

## Freeze blockers

- Dataset identities, licenses, exact files, hashes, periods, and temporal cutoffs.
- Development results and bounded choice record.
- Block-bootstrap simulation confirming the implementation.
- Complete source/tests and hosted dry-run receipt.
- Source commit/diff hash and Kaggle environment manifest.

