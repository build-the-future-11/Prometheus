# Model card — composition ensemble with calibrated intervals

## Model

The proposed model is a three-seed ExtraTrees regression ensemble over formula-only features. Each seed contributes 96 trees. A disjoint, canonical-group-isolated calibration subset scales tree dispersion and supplies normalized split-conformal scores. Each official fold produces its own checkpoint.

Baselines are a train-median predictor and standardized ridge regression with train-only inner alpha selection.

## Inputs and outputs

Input: a syntactically valid chemical formula using known element symbols, parentheses, and positive integer or decimal stoichiometry.

Outputs: predicted experimental band gap in eV, raw tree dispersion, calibrated sigma, split-conformal interval, a working probability of the configured target window, and elements unseen in proper training.

## Intended use

- Reproduce the frozen Matbench development study.
- Compare simple composition baselines.
- Rank a bounded, explicit candidate CSV for follow-up measurement planning.
- Study abstention and element-level extrapolation limits.

## Limitations and unsafe interpretations

- Composition-only predictions cannot distinguish polymorphs or phases.
- The Gaussian target-window probability is a ranking score, not a literal guarantee.
- Split-conformal coverage is marginal under exchangeability and can fail by subgroup or distribution shift.
- Element holdout is a deliberately harsh extrapolation test, not an official Matbench score.
- The historical candidate ledger is not a discovery list and does not validate stability, synthesis, safety, novelty, or utility.
- Official-fold metrics include a small upstream canonical-duplicate leakage risk disclosed in `DATA_CARD.md`.

## Training data and preprocessing

All model features are specified in `PROTOCOL.md`. Scaling for ridge is fit inside the official outer-training fold. The nonlinear model sees only proper-training rows; calibration rows are never used to fit trees. Test rows are never used for fitting or calibration.

## Evaluation

Complete metrics were not generated because the frozen no-pageout gate refused both development runs. `RESULTS.md` records the resource-blocked outcome. When a future run succeeds, metrics and subgroup results will be generated with row-level evidence under that new run directory. Do not copy a metric without its successful run ID and artifact hashes.

