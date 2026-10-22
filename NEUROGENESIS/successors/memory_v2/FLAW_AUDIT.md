# Adversarial flaw audit

| Flaw | Cause | Severity | Repair | Required evidence |
|---|---|---|---|---|
| Undefined original mechanism | Incompatible historical descriptions | Critical | Explicit delegated successor decision, preserve v1 | Decision and dated specification |
| Positive-outcome stopping | Request combines perfection with rigorous termination | Critical | Fixed budget and terminal classification including negative | Frozen protocol and all attempts |
| Generic memory reuse lacks novelty | DWM, Learn++.NSE, Condor, CEP, DynaME, conformal reuse predate this study | Critical | Narrow contribution to audited delayed-credit/update tradeoff; no novelty superiority claim | Source-backed related-work comparison |
| Weak stale-gradient comparison | v1 updates from old prediction residuals | High | Fresh-residual NLMS/Adam and RLS grids | Baseline selection and full denominator |
| Extra model bank confound | More stored coefficients and prediction work | High | All-update bank, capacity ablation, actual operation accounting | Paired control outcomes; no compute-matched claim from equal caps |
| Delayed-credit leakage | Re-evaluating an old sample with newly trained experts | Critical | Cache original predictions; unique expert IDs | Tests for birth, eviction and delayed feedback |
| Target leakage | Task generator knows hidden regimes and targets | Critical | Method API receives only features, then admitted feedback | Poisoned-future-target tests and timing audit |
| Regime oracle | Fixed switch timing could be used by policy | Critical | No regime argument, separate generated task metadata | Source/API audit; no-recurrence and delay stress |
| Inadequate diversity | v1 five seeds and one linear family | High | Fresh 20-seed condition matrix and real temporal stream | Complete raw traces; scoped conclusions |
| Repeated use of protected data | Improving after inspecting held-out scores | Critical | Source/config lock before confirmation; one phase only | Lock timestamp and hash checks |
| Gate has startup bias | New experts have no observed errors | High | Minimum eight feedback observations; no zero-loss eligibility | Hand-verifiable cold-start tests |
| FIFO forgets useful history | Fixed bounded bank can evict recurring concepts | Medium | K=1/4/8 controls, explicit FIFO events | Capacity tests and ablation outcomes |
| Fewer updates may cost more work | Bank dot products and loss scoring | High | Report coordinate operations, state bytes, RSS, wall time | Separate efficiency table |
| Numerical explosion | Unscaled distractors / long delay / RLS conditioning | High | NLMS denominator, finite checks, clipped Adam, symmetric RLS | Stress tests and retained failure artifacts |
| Resume changes information | Lost pending labels/predictions | High | Exact full state plus task/config identity | Interrupted replay equivalence |
| Statistical pseudoreplication | Treating time steps as IID | Critical | Bootstrap paired seed-level effects | Independent stats check |
| Real-data missingness changes time | Dropping missing rows compresses delay | High | Keep row clock; skip only unavailable target scoring | Missingness/timestamp tests |
| Real-data normalization leakage | Full-stream means/scales | Critical | Fixed physical scales, explicit missing indicators | Source and transform audit |
| Real-data delay is artificial | Dataset has no label-release timestamps | High | State this limitation explicitly | Paper methodology and metadata |
| New algorithm not guaranteed | Known ensemble components | High | Report empirical study and negative/mixed findings honestly | Claim-to-artifact audit |
| External publication identity | Parent Atlas dirty and no child Git | High | Separate clean Atlas branch, subtree-only commit | Remote tree and hash verification |
| Style template provenance | Neighboring arxiv.sty files are shims | Medium | Use user's refined template from provenance directory | Copy SHA-256 and PDF visual check |
| Local memory pressure | Host pageout activity | High | Bounded serial jobs with admission receipts; prefer hosted execution | Pre/post resource records |
| Hosted result confused with preparation | Notebook upload is not run completion | Critical | Retrieve output and verify source/results hashes | Saved Kaggle version and downloaded archive |
| Independent replication overstated | Same agent and same implementation | High | Clean-environment replay only; external replication remains absent | Explicit limitation |

Current status: remedies specified; implementation and experimental evidence pending. Completion is determined by actual receipts, not this checklist text.
