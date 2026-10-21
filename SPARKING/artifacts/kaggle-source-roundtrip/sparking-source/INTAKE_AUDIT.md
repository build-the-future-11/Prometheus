# Intake and adversarial idea audit

Date: 2026-10-20 (Asia/Kolkata)

## Repository and predecessor boundary

`SPARKING` began as an empty set of directories plus a generic Atlas feature
checklist. Its Git root is the parent `Atlas` checkout, whose unrelated deleted
and untracked paths are not owned by this project. The scientifically relevant
predecessor is `../ML4-Material Discovery`; its retained state is
`NOT_EVALUATED_RESOURCE_BLOCKED`. SPARKING is a successor and does not edit or
reinterpret those failures.

## Exact direction

Existing composition-only nonlinear models can interpolate well but have no
general guarantee under chemical support shift. SPARKING predicts with a ridge
component plus a histogram-gradient-boosting residual. A nearest-neighbour
support gate attenuates only the residual outside a threshold estimated from
training leave-one-out distances. The compute-matched control uses the same
features, ridge model, residual learner, data, and hyperparameters without the
gate.

## Flaw -> cause/evidence -> severity -> repair -> required evidence

| Flaw | Cause / current evidence | Severity | Repair | Acceptance evidence |
|---|---|---:|---|---|
| Novelty may be incremental | Applicability-domain distances, residual learning, and conformal prediction all pre-exist | High | Claim only the specific support-gated residual mechanism and matched element-exclusion test; do not claim invention of its ingredients | Related-work comparison plus code-level mechanism-off control |
| Distance can be meaningless in a poor descriptor space | Element fractions and periodic moments omit structure and processing | High | Standardize a fixed declared feature space; test under element exclusion; state the composition-only boundary | Per-holdout distance/gate diagnostics and failure analysis |
| The gate can suppress useful nonlinear corrections | OOD distance is not proof a residual is wrong | High | Freeze a development-only grid; require improvement on at least 4/5 protected elements and a practical macro effect | Protected paired element results |
| A simpler ridge model may be enough | Extrapolative linear descriptors can outperform nonlinear models OOD | High | Include ridge as a strong simple baseline | Ridge MAE for every holdout and official folds |
| Apparent benefit may be compute or parameter driven | A new ensemble could add capacity | Critical | Compare SPARKING to the identical ungated residual model; the gate adds no trained predictive parameters | Matched-control contract and timings |
| IID improvement could hide OOD failure, or vice versa | Random folds and element holdouts answer different questions | Critical | Make element exclusion primary and official Matbench five-fold performance a non-inferiority constraint | Both endpoint families in one frozen report |
| Official folds contain canonical-composition overlap | Predecessor audit found overlap in folds 0--3 | High | Preserve official comparability but label the leakage caveat; do not treat it as extrapolation evidence | Duplicate audit and separate element-exclusion endpoint |
| Five element scenarios are not independent materials | Rows overlap across element holdouts and n=5 is small | High | Treat elements, not rows, as the resampling unit; report every element and a bootstrap interval without overstating precision | Element-level table and interval |
| Hyperparameter search can overfit | Many gate settings can exploit development noise | High | Cap development at eight declared settings across Cu/Ag/B and freeze before Li/Zn/Ga/Se/Te | Development receipt and successor freeze digest |
| Positive-only iteration would invalidate confirmation | Redesign after protected outcomes converts confirmation into tuning | Critical | Classify valid negative/mixed outcomes honestly; materially changed ideas require a new successor | Immutable confirmatory receipt and state transition |
| Composition does not identify structure or polymorph | Multiple structures can share a formula | Critical | Exclude stability, synthesis, and discovery claims | Paper limitation and claim-evidence matrix |
| Experimental labels are heterogeneous | Matbench aggregates literature measurements | Medium | Use robust absolute-error losses and distribution-free marginal intervals; avoid aleatoric/causal interpretations | Interval coverage and dataset-card caveat |
| Local host is resource unstable | Predecessor pageouts rose during five attempts | Critical | Keep 512 MiB/no-rising-pageout stop rule and execute on a hosted Linux CPU | Admitted preflight and complete hosted receipt |
| Acquisition can drift or block automation | First SPARKING request to the canonical URL returned HTTP 403 without a user-agent | Medium | Add an explicit research-client user-agent without changing URL or digest | Downloaded artifact must match SHA-256 exactly |

## Checkpoint 1 decision

The idea is ready for a bounded development test. It is falsifiable, its
closest conceptual ingredients are known, its strongest simpler and
compute-matched controls are explicit, and the protected element outcomes are
not used for development selection.

