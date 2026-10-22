# Adversarial flaw audit

Each row is a binding research risk, not marketing copy.

| ID | Flaw | Cause / evidence | Severity | Repair | Acceptance evidence |
|---|---|---|---|---|---|
| F01 | Novelty may be incremental | Fixed share, expert tracking, concept-drift adaptation, and adaptive conformal inference already exist. | Critical | Limit the contribution to directed share allocation; compare against exact constant-share and Hedge controls; avoid SOTA language. | Literature matrix plus component-matched ablation; reviewer-style novelty audit. |
| F02 | Apparent gains may come from more experts | DDFS has a multi-timescale bank while a single RLS does not. | Critical | Primary comparator uses the identical expert bank, parameters, features, updates, and compute; only redistribution differs. | Source-level equivalence test and per-step expert-prediction equality. |
| F03 | Test leakage through online updates | Prequential forecasting observes labels after prediction; an implementation bug could consume a current label early. | Critical | API requires `predict()` immediately followed by `update(y_t)` and rejects other orderings. | Causality unit test and timestamp-level prediction ledger. |
| F04 | Seasonal period can encode privileged knowledge | A known period can substantially help energy data. | High | Give every method the same period; include seasonal-naive; declare periods before protected evaluation. | Protocol/config diff and baseline qualification table. |
| F05 | Drift trigger may merely react to noise | Fast loss can exceed slow loss without structural change. | High | Cap normalized loss, cap share, retain long-memory mass, and test stationary/noise robustness. | False-trigger rate and stationary-regime ablation. |
| F06 | RLS can become numerically unstable | Covariance recursion amplifies ill-conditioned features, especially with short forgetting. | Critical | Normalize targets/features, symmetrize covariance, validate diagonals/finiteness, bounded reset with reset counts. | Long stress test, finite predictions, reset accounting. |
| F07 | Results may be driven by outliers | Squared expert updates and raw errors can overreact. | High | Clip normalized target and aggregation loss; retain unclipped evaluation error. | Spike-corruption robustness and raw error distributions. |
| F08 | One dataset is not a general claim | Correlated series from one grid are not independent domains. | Critical | Use multiple licensed domains/datasets and report per-series, per-domain, and pooled results separately. | Cross-dataset replication matrix with no hidden denominator. |
| F09 | Timestamp errors are autocorrelated | IID confidence intervals would be invalid. | Critical | Use paired moving-block bootstrap with block length declared from seasonality; dataset/series is the top aggregation unit. | Coverage simulation and seed-fixed bootstrap receipt. |
| F10 | Development can overfit benchmark choices | Repeatedly viewing results invites researcher degrees of freedom. | Critical | Fixed revision/config budgets; append-only development ledger; freeze before final segments. | Budget ledger and signed freeze receipt predating confirmatory output. |
| F11 | Hyperparameter asymmetry can favor DDFS | Proposed method has drift gain/share settings absent from controls. | High | Small preregistered grid on development only; count every trial; tune constant share over an equal-sized grid. | Trial ledger and tuning-budget table. |
| F12 | Positive average can hide severe failures | Some series or regimes may degrade substantially. | High | Report every series, worst-case relative change, failure slices, and denominator. | Failure table and series-level forest plot. |
| F13 | Interval coverage is not guaranteed by the point model | Rolling residual quantiles under dependence are heuristic. | High | Treat intervals as secondary diagnostics; compare coverage/width; make no finite-sample coverage claim. | Explicit claim gate and calibration table. |
| F14 | Runtime advantage may be trivial at small scale | Linear models are already cheap. | Medium | Measure wall time, peak RSS where available, and operations per prediction; do not claim scalability beyond tested range. | Resource table and complexity derivation. |
| F15 | Kaggle completion is not independent replication | Same code and author on a hosted runner remains first-party execution. | Critical | Label it hosted execution; perform a clean local/second-host replay when resource admission permits. | Independent environment receipt with regenerated hashes. |
| F16 | Parent Git checkout is unsafe to commit wholesale | Parent has unrelated deletions and edits on another project branch. | Critical | Scope all staging to this directory; never reset/clean; publish only after remote/auth and diff review. | Scoped staged diff and remote tree verification. |
| F17 | “Positive results” pressure can bias the study | User objective requests positive results. | Critical | The terminal result may be supported, mixed, negative, inconclusive, or invalid; never retune a frozen test. | Complete run ledger including failures and unchanged decision rules. |
| F18 | The v1 primary comparison confounds two mechanisms | DDFS changes both the amount of shared mass and its target relative to constant fixed share. A favorable result could not be attributed to directed redistribution. | Critical | Preserve v1 and add `adaptive_share` plus `directed_fixed_share` controls in a versioned successor. | Source-level isolation tests and all four planned paired contrasts in development/confirmation. |
| F19 | The hosted source archive was not byte-deterministic | Tar members had fixed timestamps, but the gzip header used the build time. | High | Set the gzip timestamp and filename header explicitly and verify two clean builds are byte-identical. | Repeated-build hash equality on an admitted host. |
| F20 | The baseline set originally omitted stronger adaptive aggregation | Hedge/fixed share do not cover tuning-free or second-order aggregation. Revision 2 adds AdaHedge but still does not implement BOA. | High | Qualify AdaHedge on the identical bank; implement BOA before any broad best-aggregation claim, or retain the narrow fixed-share mechanism scope. | Baseline qualification record, invariance/finite-state tests, and frozen comparative results. |
| F21 | Mode-specific detector loss would break exact control matching | If each mode detects drift from its own ensemble error, differing weights cause differing future detector states; a contrast then changes both the target/rate and its trigger. | Critical | In revision 2, compute the fast/slow detector from mean capped loss over the identical expert bank, independent of ensemble weights. | Longitudinal test proving detector and per-expert loss states remain identical across all four factorial controls. |
| F22 | Independent series bootstraps discard shared grid shocks | The ten PJM regional series are contemporaneous and correlated. Resampling each series separately destroys cross-series dependence and can narrow the aggregate interval. | Critical | Use the same moving-block starts across all aligned PJM series; treat another domain as a separate top-level unit in replication. | Dependence-preservation fixture plus frozen synchronized-bootstrap receipt and timestamp-alignment check. |
| F23 | Boundary-day omissions could pass the loader | Consecutiveness was checked only between observed days; a file starting after or ending before a requested boundary could silently shorten a registered cell. | Critical | Require the first and last aggregated dates to equal the requested inclusive boundaries. | Regression test for a missing final boundary day and hosted per-file first/last-date manifest. |

## Post-execution disposition

- F02, F03, F18, F21, F22, and F23 were addressed by source-level invariants,
  hosted tests, synchronized inference, exact boundary receipts, and the retained
  factorial contrasts.
- F12 was addressed for the primary comparison: all ten per-series effects are
  retained, and every one is unfavorable.
- F15 remains open. Kaggle execution plus local standard-library recomputation is
  still first-party verification, not independent reproduction.
- F08, F14, and F20 remain open because no second domain, matched resource benchmark,
  or BOA-class comparator was executed. Those gaps limit claims rather than invalidate
  the narrow negative result.
- F17 was enforced: the negative outcome was preserved, the decision rule was not
  weakened, and the protected segment remained unopened.
- F19 was partly addressed by deterministic gzip metadata and exact hosted file
  verification. A second clean-host archive build was not performed, so cross-host
  byte reproducibility is not claimed.

