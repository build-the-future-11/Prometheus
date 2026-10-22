# Research execution status — 2026-10-21

## Outcome

Revision 2 is complete at its registered development gate. The private Kaggle CPU
run completed successfully, its source and result artifacts were verified, and the
scientific hypothesis was contradicted. Protected confirmation was therefore not
admitted. The project supports a rigorous negative development result, not a positive
performance, mechanism, generalization, efficiency, SOTA, or submission-ready claim.

## Remote execution receipt

- Private source dataset: `ryangomez010/adaptive-predictive-source-v2-20261021`
  (`ready`).
- Private kernel: `ryangomez010/adaptive-predictive-development-v2-20261021`
  (`COMPLETE`, CPU only, GPU disabled, internet disabled).
- Source archive identity: `4548ba0486279fa45706aa61412ea4efb3bd90803996f38ab708fe7a10a53ebc`.
- Hosted source verification: all 23 registered files matched the manifest.
- Hosted engineering suite: 25 of 25 tests passed before evaluation.
- Result archive identity: `742326e430bc01a15a08151edfe3c198f348e2cbfe0a17e23c1dfa34dc06e4a4`.
- Evaluation denominator: ten registered PJM series, 366 days per series, 3,660
  raw prediction rows; no protected date was parsed.

## Development results

Lower MASE is better. Equal weight is given to each of the ten series.

| Method | Mean MASE |
|---|---:|
| AdaHedge | 0.614409 |
| Hedge | 0.614465 |
| Fixed share | 0.621467 |
| Directed fixed share | 0.621916 |
| Adaptive share | 0.623639 |
| DDFS | 0.624662 |
| Uniform ensemble | 0.662893 |
| Seasonal naive | 1.026540 |

The registered primary contrast, DDFS minus fixed share, was
`+0.003195193449453197` MASE. Its synchronized seven-day moving-block bootstrap
95% interval was `[+0.0014690581194003352, +0.004907738240006376]`, the bootstrap
probability of improvement was `0.0`, and DDFS won on `0/10` series. The relative
improvement was `-0.5141%`.

The mechanism-isolating DDFS-minus-adaptive-share contrast was
`+0.001023616406604363`, with 95% interval
`[+0.0004661806514595297, +0.0015807069269040872]` and `0/10` wins. Adaptive share
minus fixed share was `+0.002171577042848835`, and directed fixed share minus fixed
share was `+0.0004485800498956668`; both were unfavorable on all ten series. DDFS
was also worse than AdaHedge by `+0.01025341509261225` MASE on all ten series.

## Independent verification

The downloaded run archive and all three inner evidence files matched their recorded
SHA-256 digests. `scripts/verify_development_v2.py` then recomputed every per-series
MAE, MASE, RMSE, bias, interval metric, aggregate MASE, and registered point contrast
directly from the raw JSONL ledger using only the Python standard library. It imported
no research-package code and reported zero issues. Bootstrap confidence intervals are
retained from the registered hosted implementation; that implementation passed its
hosted unit tests.

## Claim classification

| Claim | Final revision-2 status |
|---|---|
| Executed source identity and engineering behavior | Verified hosted: 23 source hashes; 25 tests |
| DDFS improves over constant fixed share | Contradicted in development |
| Directed redistribution adds value beyond adaptive sharing | Contradicted in development |
| Adaptive sharing improves fixed sharing | Contradicted in development |
| DDFS is competitive with strong online aggregation | Contradicted against AdaHedge in development |
| Confirmatory benefit | Not evaluated; protected confirmation not admitted |
| Cross-domain generalization or SOTA | Unsupported |
| Independent reproduction | Not performed |

## Scientific stop decision

Scheduling protected confirmation after both advancement gates failed would violate
the frozen decision rule and spend protected data on a hypothesis that did not survive
development. No confirmatory or cross-domain claim has been made. A future attempt
must be a separately versioned successor with a new rationale and preregistration;
this negative result and all its receipts remain immutable.
