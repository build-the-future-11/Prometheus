# Related work and claim boundary

Checked against primary sources on 2026-10-21. This is a mechanism-level comparison, not evidence that every competing algorithm was reproduced. Only the methods listed as implemented participate in the registered v4 comparisons.

| Primary source | Relevant result | Relationship to v4 | Matched implementation |
|---|---|---|---|
| [Kolter and Maloof, Dynamic Weighted Majority, JMLR 2007](https://www.jmlr.org/papers/v8/kolter07a.html) | Adds, reweights, and removes online experts in response to performance under concept drift | Performance-controlled expert pools are established prior art; v4's FIFO frozen bank and delayed squared-loss gate are narrower | No |
| [Joulani, Gyorgy and Szepesvari, ICML 2013](https://proceedings.mlr.press/v28/joulani13.html) | Formalizes online learning with delayed feedback and shows qualitatively different regret effects in adversarial and stochastic settings | The four- or sixteen-step release clock is an information constraint; v4 makes no regret or optimality theorem | No |
| [Goel, Bruce and Bernstein, variable-direction-forgetting RLS, 2020](https://arxiv.org/abs/2003.03523) | Explains loss of robustness under uniform forgetting without persistent excitation and motivates direction-selective forgetting | Directly explains the predecessor's covariance failure; fixed-ridge v4 is a simpler changed objective, not the same algorithm | No |
| [Lai and Bernstein, SIFt-RLS, 2024](https://arxiv.org/abs/2404.10844) | Forgets only in the regressor information subspace | A stronger rank-aware alternative exists; it is outside the one-mechanism frozen budget and v4 cannot claim superiority over it | No |
| [Chen, Beijing PM2.5, UCI 2015](https://archive.ics.uci.edu/dataset/381/beijing%252Bpm2%252B5%252Bdata) | Hourly pollutant and meteorological record with documented missing PM2.5 values, 2010--2014, CC BY 4.0 | Supplies the preselected chronological transfer stream; v4 uses 2013 warmup and 2014 evaluation without target lags | Dataset only |

The implemented confirmation comparators are persistent-ridge RLS, NLMS/Adam-selected development controls, every-third-label periodic solves, FIFO replay, k-nearest-neighbor cache, frozen prediction, an always-update bank, a one-slot bank, a uniform bank, and a no-bank ablation. These controls test simpler explanations inside the registered budget. They are not substitutes for reproducing DWM, variable-direction forgetting, or SIFt-RLS.

## Narrow contribution under test

V4 asks whether maintaining a fixed ridge information floor eliminates the predecessor's numerical failure while a delayed-credit FIFO predictor bank retains a useful prediction-error/weight-solve tradeoff. Persistent regularization, expert reuse, delayed feedback, and solve suppression are all established ingredients. A defensible result is therefore a fully specified and reproducible interaction study, including negative conditions and numerical failures—not a claim of a new general-purpose learning principle or state of the art.

## Theoretical boundary

The registered information matrix satisfies

\[
A_n=\rho I+\sum_{i=1}^{n}\lambda^{n-i}x_i x_i^\top,
\]

so its smallest eigenvalue is at least \(\rho\) in exact arithmetic. Under bounded inputs and targets this yields finite conditional eigenvalue and coefficient bounds. It does not establish low prediction error, calibrated uncertainty, regret optimality, robustness to arbitrary distributions, or floating-point infallibility. The implementation therefore checks Cholesky pivots and a scaled backward residual, and any failed cell invalidates support.

## Data boundary

Beijing is one chronological stream, not a collection of independent datasets. The four-hour label delay is imposed by the study rather than documented by UCI. Contemporaneous weather availability is assumed. Seven evaluation blocks are descriptive; no IID confidence interval or cross-dataset generalization claim is permitted. The already exposed UCI Air Quality stream remains regression evidence only.
