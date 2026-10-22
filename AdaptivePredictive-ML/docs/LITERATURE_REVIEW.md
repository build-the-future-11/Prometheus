# Literature review and novelty boundary

## Scope and search record

This review was refreshed on 2026-10-21 from original papers, proceedings pages,
and author-hosted manuscripts. It concerns causal online aggregation under drift,
not generic offline time-series forecasting. Search terms covered fixed share,
tracking/adaptive regret, Markov expert priors, online electricity-load aggregation,
forgetting-factor RLS, and recent online drift-adaptive forecasting.

## What is already established

| Work | Established contribution | Consequence for this project |
|---|---|---|
| Herbster & Warmuth (1998), [Tracking the Best Expert](https://doi.org/10.1023/A:1007424614876) | Fixed-share and variable-share updates for tracking changing experts with regret guarantees. | Neither weight sharing nor loss-dependent sharing is novel here. |
| Adamskiy et al. (2016), [A Closer Look at Adaptive Regret](https://www.jmlr.org/papers/volume17/13-533/13-533.pdf) | Exact adaptive-regret analysis and optimality results for fixed share. | A heuristic variant must not imply a stronger regret result without a proof. |
| Mourtada & Maillard (2017), [Efficient Tracking of a Growing Number of Experts](https://proceedings.mlr.press/v76/mourtada17a.html) | MarkovHedge expresses general time-varying transition matrices over experts. | Non-uniform or time-varying redistribution is part of a broader known Markov-prior framework. RDDFS can at most claim a particular causal transition policy unless it proves a new bound. |
| Devaine et al. (2013), [Forecasting Electricity Consumption by Aggregating Specialized Experts](https://doi.org/10.1007/s10994-012-5314-7) | Fixed-share/specialist aggregation was already applied to real electricity-load forecasting with online tuning. | Electricity-load aggregation is not a new application category; comparison to strong online aggregation is mandatory. |
| Wintenberger (2017), [Optimal Learning with Bernstein Online Aggregation](https://doi.org/10.1007/s10994-016-5592-6) | Second-order online aggregation and adaptive learning rates with fast-rate guarantees in stated settings. | Plain Hedge and fixed share alone are not sufficient to represent the strongest aggregation literature. A feasible second-order comparator is needed before a broad competitiveness claim. |
| de Rooij et al. (2014), [Follow the Leader If You Can, Hedge If You Must](https://www.jmlr.org/papers/v15/rooij14a.html) | AdaHedge adapts the Hedge learning rate from the observed mixability gap without advance knowledge of horizon or loss range. | A tuning-free comparator is feasible on the exact expert bank and is included in revision 2; this does not substitute for BOA in any second-order-optimality claim. |
| Ba et al. (2012), [Adaptive Learning of Smoothing Functions](https://proceedings.neurips.cc/paper/2012/hash/e2c420d928d4bf8ce0ff2ec19b371514-Abstract.html) | Forgetting-factor RLS and an adaptive forgetting factor were evaluated for electricity load. | Multi-timescale RLS and adaptive forgetting are prior art; the expert bank is a baseline substrate, not the contribution. |
| Yuan & Lamperski (2020), [Trading-Off Static and Dynamic Regret in Online Least-Squares and Beyond](https://doi.org/10.1609/aaai.v34i04.6149) | Dynamic-regret analysis for forgetting-factor RLS/online Newton methods. | Any theoretical discussion must distinguish this project’s empirical transition heuristic from established RLS regret results. |
| Zhang et al. (2023), [OneNet](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html) | A substantially larger online ensemble for time-series concept drift, evaluated on multivariate benchmarks. | OneNet is relevant modern context, but it is not compute- or task-matched to the present univariate CPU study and must not be treated as a direct head-to-head result without reproduction. |
| Chen et al. (2025), [LEAF](https://doi.org/10.24963/ijcai.2025/542) | Meta-learned extrapolation and adjustment for macro/micro drift with real load benchmarks. | The broader online-forecasting field is far beyond linear expert aggregation; claims should emphasize resource-light causal behavior rather than SOTA forecasting. |
| Hyndman & Koehler (2006), [Another Look at Measures of Forecast Accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001) | MASE as a scale-free accuracy measure. | Training-only seasonal-naive scaling is appropriate, but the exact denominator and per-series weighting must remain frozen and visible. |

## Defensible novelty statement

The current implementation is **not** a novel theory of fixed share, expert
tracking, RLS, adaptive forgetting, or concept-drift forecasting. Its only plausible
incremental contribution is an empirical test of a specific causal transition policy:
after observing the previous outcome, it uses ensemble loss deterioration to control
the share rate and a geometric combination of memory length and relative recent expert
loss to set a non-uniform redistribution target.

That statement is still a hypothesis. MarkovHedge already covers general transition
matrices, and variable-share methods already vary sharing using loss. A publication
claim therefore requires all of the following:

1. mechanism-isolating controls for share amount and share direction;
2. a strong adaptive/second-order aggregation comparator when feasible;
3. multiple real domains or an explicitly narrow single-domain claim;
4. a frozen causal protocol with no protected-test inspection;
5. consistent practical improvement, uncertainty, and failure slices; and
6. a clear resource-efficiency result if accuracy is not superior.

## Required comparison set

| Priority | Comparator | Why it is required | Current status |
|---|---|---|---|
| 1 | Constant uniform fixed share on the same RLS bank | Overall mechanism-off control | Evaluated; DDFS was worse by +0.003195 MASE, 0/10 series wins |
| 2 | Adaptive share rate with uniform target | Isolates directed targeting | Evaluated; DDFS was worse by +0.001024 MASE, 0/10 wins |
| 3 | Constant share rate with directed target | Isolates rate adaptation | Evaluated; DDFS was worse by +0.002747 MASE, 0/10 wins |
| 4 | Hedge and uniform average on the same bank | Separates sharing from ordinary reweighting | Evaluated; Hedge outperformed DDFS and uniform |
| 5 | Seasonal naive | Transparent task baseline | Evaluated; every ensemble outperformed seasonal naive |
| 6 | AdaHedge on the identical expert bank | Strong tuning-free aggregation baseline | Evaluated; AdaHedge was best and beat DDFS by 0.010253 MASE |
| 7 | Second-order online aggregation such as BOA | Represents stronger second-order methodology | Not implemented; required only for a broad best-aggregation claim, not the narrow fixed-share mechanism test |
| 8 | Modern deep online forecasting (OneNet/LEAF class) | Establishes broader field position | Literature-only context; direct reproduction is outside the current matched CPU budget unless separately authorized and resourced |

## Final revision-2 conclusion

The proposed novelty remains incremental, and its empirical mechanism failed the
registered development test. Adaptive share rate and directed targeting were each
unfavorable in isolation; their combination was worse than constant fixed share on
all ten series and worse than AdaHedge by 0.010253 MASE. The defensible contribution
is a reproducible negative result and a mechanism-isolating design, not a new regret
theory, a positive forecasting method, or a SOTA claim. Protected confirmation was
not admitted after the advancement gates failed.
