# Model card: SPARKING v0.1

## Intended use

SPARKING tests whether a support-distance gate can make a composition-only
nonlinear residual model safer under declared element exclusions. It is a
research model for one public band-gap dataset, not a materials-discovery or
deployment system.

## Inputs and outputs

Input formulas are parsed without silent correction. Each is mapped to 118
elemental fractions and 14 global statistics: element count, fraction entropy,
fraction extrema and L2 norm, and weighted moments/ranges of atomic number,
period, and group. The output is a non-negative point estimate in eV. A
separate split-conformal procedure produces constant-radius 90% marginal
intervals for the official-fold compatibility study.

## Mechanism

1. Fit standardized ridge regression to the target.
2. Fit a histogram gradient-boosting regressor to the ridge residual with
   absolute-error loss.
3. Standardize the same feature vector for distance computation.
4. Estimate a support threshold from a declared quantile of training
   leave-one-out nearest-neighbour distances.
5. Preserve the full residual correction in support and attenuate it as a
   power law outside support.

The compute-matched mechanism-off control uses the exact same ridge and
residual models with gate value one everywhere.

## Training contract

- Ridge alpha candidates: 17 logarithmically spaced values from 1e-4 to 1e4.
- Residual learner: 180 iterations, learning rate 0.055, 31 leaves,
  minimum leaf size 12, L2 regularization 0.1, seed 20261020.
- Development gate grid: support quantile {0.80, 0.90, 0.95, 0.99} crossed
  with power {1, 2}; exactly eight candidates.
- Selection: lowest macro-MAE across Cu/Ag/B; candidates within 0.002 eV use
  lower power and then higher support quantile.

## Limitations

Nearest-neighbour distance depends on the chosen standardized feature space and
is not a probability of correctness. Split conformal intervals guarantee only
marginal coverage under exchangeability, not conditional or shifted-domain
coverage. Element exclusions can create chemically severe shifts and overlapping
test scenarios. No prediction should substitute for structural calculation or
measurement.

## Current status

`NOT_EVALUATED`. The local development attempt was `FAILED_RETAINED` after the
pageout counter increased by eight during featurization. No model was fitted and
no performance claim is authorized from that run.

