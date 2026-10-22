# Protocol draft v0.3 — development revision 2

This document supersedes draft v0.2 for revision-2 development only. Draft v0.2
and protocol `pjm-daily-development-v1` remain historical records. Nothing in this
draft authorizes protected confirmation.

## Causal online formulation

At time (t), all methods observe only (y_{<t}). Expert (i), with forgetting
factor \(\lambda_i\), emits a residual forecast \(\hat r_{i,t}\) from the shared
RLS feature vector. With seasonal-naive base \(b_t=y_{t-s}\) and causal robust
scale \(c_t\), its forecast is

\[
\hat y_{i,t}=b_t+c_t\,\operatorname{clip}(\hat r_{i,t},-8,8).
\]

The aggregation weights (w_t) produce
\(\hat y_t=\sum_i w_{i,t}\hat y_{i,t}\). Only after that forecast is recorded is
(y_t) revealed. Every aggregation method receives the identical capped expert
loss vector

\[
\ell_{i,t}=\min\!\left(\frac{|\hat y_{i,t}-y_t|}{c_t},8\right).
\]

Experts are then updated on the same feature/target pair. Thus aggregation choices
cannot alter the expert bank.

## Common drift signal

The detector must be independent of aggregation weights. Define

\[
d_t=K^{-1}\sum_{i=1}^K \ell_{i,t},\quad
F_t=(1-\rho_f)F_{t-1}+\rho_f d_t,\quad
S_t=(1-\rho_s)S_{t-1}+\rho_s d_t,
\]

and

\[
z_t=\operatorname{clip}\!\left(\frac{F_t-S_t}{\max(S_t,0.05)},0,1\right).
\]

The registered rates are \(\rho_f=0.20\) and \(\rho_s=0.02\). The outcome at
time (t) can affect only the weights used at (t+1).

## Posterior and share mechanisms

Fixed-rate Hedge first forms

\[
q_{i,t+1}=\frac{w_{i,t}\exp(-\eta\ell_{i,t})}
{\sum_j w_{j,t}\exp(-\eta\ell_{j,t})}.
\]

The share rate is either constant \(\alpha_0\) or adaptive:

\[
\alpha_t=\operatorname{clip}(\alpha_0+\gamma z_t,\alpha_0,\alpha_{\max}).
\]

The uniform target is (u_i=1/K). For the directed target, let

\[
m_i\propto 1-\lambda_i+10^{-3},\qquad
p_{i,t}\propto \exp\!\left[-\frac{F_{i,t}/\max(S_{i,t},0.05)}{0.5}\right],
\]

where (F_{i,t}) and (S_{i,t}) are fast and slow EWMAs of the same expert loss.
Normalize \(r_{i,t}\propto\sqrt{m_i p_{i,t}}\), then define

\[
g_{i,t}=(1-z_t)u_i+z_t r_{i,t}.
\]

The next weights are

\[
w_{i,t+1}=(1-\alpha_t)q_{i,t+1}+\alpha_t h_{i,t},
\]

with target (h_t=u) or (g_t) according to the registered mode.

| Mode | Share rate | Target |
|---|---|---|
| `fixed_share` | \(\alpha_0\) | uniform (u) |
| `adaptive_share` | \(\alpha_t\) | uniform (u) |
| `directed_fixed_share` | \(\alpha_0\) | directed (g_t) |
| `ddfs` | \(\alpha_t\) | directed (g_t) |

This (2\times2) design identifies the empirical main effects and interaction of
rate adaptation and direction under identical experts and detector state. It does
not by itself prove a regret guarantee or causal generalization beyond the registered
streams.

## Strong aggregation control

`adahedge` uses the same expert losses but chooses its learning rate from the
cumulative mixability gap, following de Rooij et al. (2014). It has no tuned `eta`
in this study. It is a stronger tuning-free comparator, not a drift-directed share
ablation and not a substitute for a BOA comparison in a second-order-optimality claim.

## Estimand and decisions

For every registered series (d), evaluation MASE uses the seasonal-naive scale
computed only on its warm-up/training segment. The primary estimand remains the
equal-series mean of timestamp-paired `ddfs - fixed_share` scaled absolute errors.
Overall support requires an effect at most (-0.01) MASE and at least one-percent
relative improvement. A directed-mechanism statement additionally requires a
favorable `ddfs - adaptive_share` effect that is not reversed in a majority of
registered series. All other factorial contrasts and AdaHedge are reported without
selective omission.

Uncertainty uses a paired moving-block bootstrap with synchronized block starts
across the ten aligned PJM series, preserving common temporal shocks before
equal-series aggregation. Block length is seven, with 5,000 repetitions and seed
20261021. Development is selection evidence only;
no development interval can support the final empirical claim.

## Complexity

With (K) experts and RLS feature dimension (d), each timestamp costs
\(O(Kd^2)\) time and \(O(Kd^2+W)\) memory, where (W) is the retained interval-error
window. Aggregation itself is (O(K)). No lower runtime or memory claim is allowed
without matched measurements.
