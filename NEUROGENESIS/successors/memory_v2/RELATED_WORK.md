# Prior-art and claim boundary (checked 2026-10-21)

This study does not claim invention of expert reuse, drift-aware ensembles, memory storage, or skipping parameter updates. It tests one explicit combination under strict delayed-label timing and bounded memory. The fundamental gap asserted here is an evidence gap in this project, not proof that the research field has never studied the problem.

| Primary source | Relevant mechanism | Consequence for NEUROGENESIS |
|---|---|---|
| [Kolter and Maloof, Dynamic Weighted Majority, JMLR 2007](https://www.jmlr.org/papers/v8/kolter07a.html) | Creates/removes weighted experts using online performance | Dynamic expert pools and performance-based routing are established |
| [Joulani, Gyorgy and Szepesvari, ICML 2013](https://proceedings.mlr.press/v28/joulani13.html) | General online learning with delayed feedback; regret effects differ between stochastic/adversarial settings | Delay is an information constraint, not an implementation detail; our empirical linear study gives no general regret theorem |
| [Eliades and Papadopoulos, COPA 2025](https://proceedings.mlr.press/v266/eliades25a.html) | Reuses earlier models after conformal drift detection and relevance/performance tests | Recurrent model retrieval is prior art; our EW error gate is not a conformal detector and offers no coverage guarantee |
| [Zhan et al., Continuous Evolution Pool, arXiv:2506.14790v3, 2026 revision](https://arxiv.org/html/2506.14790v3) | Forecaster pools, retrieval, evolution, elimination and gradient abandonment | Even reuse plus skipped gradients is not a broad novelty claim; compare at mechanism level and disclose that their full neural implementation is not reproduced |
| [Hong et al., Dynamic Multi-period Experts, WWW 2026](https://arxiv.org/html/2603.09062v1) | Period-specific experts, closed-form regression adaptation and dynamic gating | Lightweight regression experts and non-gradient adaptation also predate this study |

## Narrow defensible claim

A reproducible, transparent test of whether delayed prediction-time credit and a finite FIFO bank yield a useful accuracy/update tradeoff against fresh-residual NLMS, Adam, RLS, cache and replay controls. The selected method is a study candidate, not an established new general-purpose model. No claim of superiority over DWM, CEP, DynaME, conformal reuse, or full state-of-the-art systems is allowed without matched implementations and data.

## Data source

[UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) documents sensor cross-sensitivity, drift, and -200 missing markers. The landing page lists 9,358 observations and describes March 2004-February 2005. The downloaded CSV actually contains 9,357 timestamped rows from 2004-03-10 18:00 through 2005-04-04 14:00; the executed file hash and timestamps take precedence for this study. The page shows CC BY 4.0 plus a legacy research-only notice; this use is noncommercial research and retains attribution. We impose four hours of label delay; the data do not establish actual feedback release times.

## Why memory may fail

Loss credit itself is delayed. Stored predictors cannot identify a hidden sign change before informative feedback arrives. A finite FIFO bank can evict useful concepts, and old scores can misroute predictions after a change. Skipping active updates can prevent learning the next novel regime. RLS or even simple NLMS may exploit the linear task structure more efficiently. Every one of these alternatives is compatible with a valid negative result.
