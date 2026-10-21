# Frozen scientific protocol

The machine-readable controlling protocol is `protocols/development-v1.json`.
This prose version explains its scientific meaning; a conflict is an error and
must be resolved by a new version before execution.

## Precise statement

Existing composition-only nonlinear predictors struggle under chemical support
shift. We propose support-gated residual learning because a regularized linear
component can extrapolate through declared periodic descriptors while nonlinear
corrections are most defensible near observed compositions. The mechanism
predicts lower leave-one-element-out MAE than the identical ungated residual
control without more than 0.02 eV degradation on official Matbench MAE.

## Development

The development budget is eight gate settings and three element scenarios:
Cu, Ag, and B. All other architecture and optimizer settings are fixed. The
selection rule is deterministic and recorded in the protocol. The development
run emits a freeze candidate but cannot execute the protected study.

## Confirmation

After a successful manifest-verified development run, a new immutable
`protocols/confirmatory-v1.json` must bind the selected gate, source revision,
development artifact digest, Li/Zn/Ga/Se/Te scenarios, official folds, endpoints,
and decision rule. Success requires all of:

1. Macro element-exclusion MAE improvement over ungated > 0.01 eV.
2. Positive improvement in at least four of five element scenarios.
3. Official-fold MAE degradation relative to ungated <= 0.02 eV.

The one-sided exact sign-permutation result and an element-level bootstrap
interval are descriptive because five overlapping scenarios provide limited
inferential resolution. Negative, mixed, inconclusive, and invalid outcomes are
terminal possibilities; protected outcomes cannot be used to retune v1.

## Resource and failure policy

Every numerical run has a 512 MiB peak-RSS ceiling and requires no increase in
the host pageout/swapout counter. Any refusal is retained with stack trace,
resource observations, receipt, and artifact manifest. Existing run IDs are
never overwritten. Partial cells are engineering evidence only.

