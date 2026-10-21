# SPARKING v2 frozen successor protocol

Frozen on 2026-10-21 before any v2 outcome-bearing development execution.
The controlling machine-readable protocol is
`protocols/development-v2.json`; this document explains its scientific intent.

## Question and contribution

SPARKING v1 showed that nearest-neighbour distance was a weak and nearly
inactive proxy for residual-model harm. V2 tests a narrower replacement: can a
risk model trained only from pseudo-element exclusions inside the current
training data identify when a strong direct composition model should be shrunk
toward a regularized linear model?

The proposed contribution is the evaluation protocol and the error-aware convex
gate, not CatBoost, Magpie descriptors, ridge regression, or element holdouts
individually. Matbench supplies the pinned experimental-band-gap dataset and
ordinary five-fold comparability endpoint. Element exclusion is the primary
shift endpoint. Composition-only predictions remain unable to identify crystal
structure, stability, synthesizability, safety, or discovery.

## Fixed representation and models

Every model receives the same target-independent representation: the v1
elemental fractions and periodic summary statistics plus Magpie elemental
property statistics and stoichiometric features. Missing values are imputed
from the training set only.

The primary direct model is a five-seed CatBoost regression ensemble with a
fixed 1,200-iteration GPU budget. Ridge, corrected squared-error histogram
boosting, and a three-seed ExtraTrees ensemble are strong controls. Constant
ridge/CatBoost blends at direct weights 0.25, 0.50, and 0.75 distinguish
adaptive risk gating from generic shrinkage.

For each outer development element, the risk calibrator is constructed only
inside that outer training set. Eleven declared pseudo elements are removed one
at a time. Ridge and the full CatBoost ensemble predict each pseudo-element
holdout, producing out-of-element predictions. A fixed histogram boosting
model learns the difference between CatBoost absolute error and ridge absolute
error from nine declared risk diagnostics. Each pseudo element has equal total
weight. The final gate preserves CatBoost when predicted harm is non-positive
and exponentially shrinks toward ridge when predicted harm is positive.

Four temperatures, 0.05, 0.10, 0.20, and 0.40 eV, are the entire development
grid. Selection minimizes macro-MAE across Cu, Ag, and B; candidates within
0.005 eV choose the largest temperature, which is the least aggressive gate.

## Development stop rule

Protected confirmation is admitted only if all conditions hold:

1. Selected-gate macro-MAE improves over the identical ungated CatBoost
   ensemble by more than 0.02 eV.
2. At least two of Cu, Ag, and B improve.
3. The gate is active on at least 1% and at most 90% of every development
   holdout.
4. CatBoost predictions are nonconstant and all eleven pseudo exclusions
   complete with finite, manifest-valid evidence.

A failure is a scientific development result and terminates v2. It does not
authorize another search over the protected Li/Zn/Ga/Se/Te outcomes.

If development passes, a separately frozen `confirmatory-v2.json` must bind the
selected temperature, source commit, protocol digest, data digest, and
development artifact digest. Confirmation must then improve macro-MAE by more
than 0.01 eV, improve at least four of five protected elements, and degrade
ordinary Matbench MAE by no more than 0.02 eV. Protected results may be positive,
mixed, negative, or invalid; none may be used to retune v2.

## Evidence contract

The private hosted run must use a real Kaggle GPU and record its identity,
driver, memory, package freeze, source archive digest, exact source inventory,
resource observations, every seed prediction, every pseudo-exclusion row,
per-example risk diagnostics, gate weights, raw predictions, aggregate metrics,
and a content manifest. A separate local verifier recomputes all reported
metrics from downloaded arrays before any research-state or manuscript update.

GPU CatBoost can be nondeterministic because floating-point reduction order can
vary, so five seeds and retained member predictions quantify the executed
variation instead of claiming byte-identical training. The relevant primary
references are the [Matbench benchmark paper](https://arxiv.org/abs/2005.00707),
the [Magpie composition representation paper](https://doi.org/10.1038/npjcompumats.2016.28),
and the [CatBoost GPU processing documentation](https://catboost.ai/docs/en/references/training-parameters/processing-unit).
