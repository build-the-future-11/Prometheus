# Pre-evaluation adversarial review

| Flaw | Cause and severity | Repair or limitation | Required acceptance evidence |
|---|---|---|---|
| V3 covariance breakdown | Vanishing regularization under weak excitation; critical | New fixed-ridge objective with direct Cholesky; preserve v3 failures | Independent batch agreement, long degenerate stream and full chronological attempts |
| Exception-free does not mean stable | V3 QR diagnostic produced extreme coefficients; critical | Mathematical information floor plus solve residual checks | All full streams complete with finite predictions and reported errors |
| Periodic updates may explain savings | Fewer solves can help without a memory bank; major | Add every-third-label solve to development and confirmation | Report periodic accuracy, sufficient-statistic updates, solves, operations and runtime |
| Known components | Memory, gating and regularization predate this project; major | Claim a narrow mechanism/evidence study | Explicit prior-art comparison, no SOTA or broad invention claim |
| Adaptation clock confound | Gate skips forgetting and information as well as weights; major | State accepted-label objective explicitly | All-update and no-bank ablations; no isolated memory-causality claim |
| Fresh test compromised by prior outcomes | Air Quality already failed v3; major | Exposed regression label; separately choose Beijing before outcomes | Freeze hashes before Beijing run, no post-outcome retuning |
| Synthetic-specific advantage | Simple recurring linear maps; major | Novel-regime, noise, delay, distractor, dimension and horizon tests | Complete seven-condition matrix, including unfavorable conditions |
| Real data mismatch | Single stream, artificial delay, weather availability assumption; major | Honest fixed transform and chronological split | Full traces; persistence control; no IID dataset-level inference |
| Absolute gate scale sensitivity | Squared-loss ceiling depends on target units; major | Fix physical scales and thresholds before evaluation | Report transfer without adapting ceiling to observed test performance |
| Fixed ridge bias | Penalizes informative coefficients and intercept; moderate | rho=1 ex ante, no tuning after outcome | Accuracy versus stronger selected conventional baseline |
| Dense solve cost | O(p^3) Cholesky per accepted update; major | Report total operation estimates, state and wall time | No speed claim from solves alone; limits stated for larger dimensions |
| Delayed credit and eviction | Stale losses, no information at hidden switch; major | Prediction-time tickets, unique FIFO IDs, minimum credit | Poisoned-future-label tests and exact trace audit |
| Multiple comparisons | 22 development settings and several controls; major | Separate fresh confirmation; one primary endpoint | One fixed bootstrap analysis; secondary comparisons descriptive |
| Missing/failed cells | Could hide bad seeds or overflow; critical | Immutable failed attempts, complete expected denominator | Coverage/hashes/replay; failures forbid support |
| External reproducibility | Same-author replay is not independent; major | Separate local, hosted and external evidence | Kaggle output archive and independent reproduction remain distinct gates |
| Resource failures | Constrained host, disk and memory; major | Prephase admission and bounded sequential execution | Retained refusals; no reduced thresholds or silent retries |

Selected direction: persistent-ridge delayed-credit memory. Alternative considered: directional/subspace forgetting as in SIFt-RLS. The fixed-ridge direction is chosen because it has a short auditable objective and directly targets the observed loss of an information floor, while adding one mechanism under the bounded budget. This is a defensible implementation choice, not evidence that it outperforms directional forgetting. A matched SIFt-RLS implementation is outside the current budget; no comparison claim is allowed.

Sources: [Goel, Bruce and Bernstein (2020)](https://arxiv.org/abs/2003.03523), [Lai and Bernstein (2024)](https://arxiv.org/abs/2404.10844). Prior memory and delayed-feedback comparisons remain in the immutable v3 RELATED_WORK.md.
