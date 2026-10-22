# SPARKING v3 frozen development protocol

## Scientific purpose

V2 established a valid negative result for sample-level harm gating. It also produced a secondary development observation: the unweighted Extra Trees ensemble was 0.050862 eV better in element-macro MAE than the matched CatBoost ensemble on Cu, Ag, and B. V3 does not reinterpret that observation as confirmatory evidence. It uses it to motivate a separately versioned hypothesis about training-distribution imbalance.

V3 tests whether target-independent inverse-frequency weighting of training compositions improves a randomized tree ensemble under leave-one-element-out shift. For a training composition `i`, the raw weight is

`sum_e x_ie * (n_e / n_train)^(-alpha)`,

where `x_ie` is the stoichiometric fraction of element `e`, `n_e` is its occurrence count in the outer training set, and `alpha` is fixed to 0.0, 0.5, or 1.0. Weights are normalized, clipped to [0.25, 4.0], and renormalized to mean one. Alpha 0.0 is exactly the unweighted matched control. No target enters the weighting rule.

## Fresh development panel

The development elements are O, S, In, Ge, Sb, Ba, Co, and F. They were selected before any v3 target outcomes using occurrence counts only. The panel spans 112 to 1,090 rows and multiple chemical roles. It excludes prior outer development elements Cu, Ag, and B; all v2 pseudo-exclusion elements; and the protected Li, Zn, Ga, Se, and Te outcomes.

Each outer holdout fits the following models on rows without the held-out element:

- Frequency-balanced Extra Trees at alpha 0.0, 0.5, and 1.0, three seeds and 900 trees per seed.
- Mean and median aggregation across the three independently seeded forests.
- Three-seed GPU CatBoost and its single-seed control.
- Squared-error histogram gradient boosting.
- Standardized ridge regression.

The selected v3 candidate is the nonzero-exponent Extra Trees variant with minimum macro MAE. A 0.003 eV tie band selects the smaller exponent, then mean aggregation. This rule limits unnecessary weighting and avoids choosing a more variable median aggregate when performance is practically tied.

## Admission and falsification

The v3 mechanism is admitted to protected confirmation only if all conditions hold:

- Macro MAE improves by more than 0.01 eV over the aggregation-matched alpha 0.0 control.
- Macro MAE improves by more than 0.02 eV over the CatBoost ensemble.
- At least five of eight elements improve over the matched unweighted control.
- Every element completes, predictions and metrics are finite, weights are nondegenerate, and the artifact manifest verifies.

Failure of any condition yields `DEVELOPMENT_NOT_SUPPORTED`. A CatBoost comparison alone cannot rescue failure against the unweighted Extra Trees control.

## Adversarial flaw audit

| Flaw | Cause | Severity | Repair in v3 | Required evidence |
|---|---|---:|---|---|
| V2 gate did not transfer | A flexible sample-level harm model fit pseudo shifts but ranked outer harm weakly | Critical | Replace target-dependent harm prediction with a fixed target-independent weighting rule | Improvement over alpha 0.0 on a fresh panel |
| Extra Trees advantage may be panel-specific | Cu, Ag, and B were inspected during v2 | Critical | Use eight fresh nonprotected outer elements | Complete per-element predictions and macro result |
| Weighting may only add effective compute | Weighted and unweighted forests could differ in stochastic work | High | Use identical features, trees, seeds, and aggregation for alpha 0.0 and nonzero alpha | Configuration and raw seeded predictions |
| Rare-element weights may become extreme | Sparse elements yield large inverse frequencies | High | Clip to [0.25, 4.0], renormalize, and retain effective sample size | Per-holdout weight diagnostics |
| Overlapping compositions violate row independence | One row can occur in several element holdouts | High | Treat elements as scenario units; cluster by row identity in any bootstrap | Retained row indices |
| Median aggregation may be opportunistic | Mean and median are both evaluated | Medium | Freeze both and a deterministic tie rule before outcomes | Candidate table and selected identity |
| Fresh panel is not a random element sample | Counts and chemical diversity drove selection | Medium | Restrict inference to the declared panel and protected protocol | Scope-correct result language |
| A successful development result could still fail confirmation | Development permits selection | Critical | Keep Li, Zn, Ga, Se, and Te sealed until a new confirmatory freeze | Explicit protected-outcome flags and absence audit |

## Execution and evidence

The private Kaggle GPU launcher installs the pinned environment, verifies the source archive byte-for-byte, runs the full test suite, executes the frozen matrix, runs an independent metric verifier, and exports the complete run directory with hashes. Failed runs and partial artifacts must be retained under a distinct run identity.
