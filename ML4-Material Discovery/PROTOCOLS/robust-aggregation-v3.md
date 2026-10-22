# Robust aggregation v3: frozen development protocol

Date: 2026-10-21. Parent: development-v1 and the negative quantile-composition-v2.
This is DEVELOPMENT on previously inspected public labels. It is not confirmatory.

## Question, mechanism, and falsifier

The historical ensemble averages tree leaf means despite an MAE endpoint and a
zero-heavy target. We test whether the median of the *same tree predictions*
reduces absolute error. This is a robust aggregation audit, not a new tree
algorithm, a conditional-quantile estimator, or a state-of-the-art claim. Median
aggregation is not guaranteed to estimate the conditional response median.
Falsifier: median aggregation does not reduce MAE by at least 0.01 eV relative
to mean aggregation in each of three prespecified grouped split repetitions.
The strongest control uses identical trees, features, fit rows, and randomness;
only aggregation changes. RMSE, nonzero-target error, and ranking can worsen.

## Inputs and independent units

Use only the hash-bound Matbench experimental gap artifact in the existing data
contract. Collapse each canonical composition to one representative formula,
the arithmetic mean of its recorded targets, and a retained list of all source
IDs and original labels. This yields 4,601 units, including all three conflicting
duplicate groups. No new official leaderboard result is implied. Formula
features remain the original 118 fractions plus ten stoichiometric/atomic-number
descriptors. Features contain no target, structure, or measurement metadata.

For each seed 20261021, 20261022, and 20261023, order canonical groups by SHA-256
of seed:canonical and assign round-robin to five outer folds. Split each outer
training pool into proper-training/calibration by a second label-blind hash
with seed+731 and 20% calibration. Save exact membership for every cell.
All models see the same proper-training rows. No outer label is used in fitting,
calibration, model selection, ranking scores, or hyperparameter choice.

## Models, controls, and numerical procedures

The proposed point model is the median of 288 ExtraTrees leaf-mean predictions:
three seeds 7,17,29, 96 trees each, depth 22, minimum leaf 2, feature fraction .7,
one worker, no bootstrap. Mechanism-off control is the mean of these same trees.
Other controls: first-seed 96-tree median (ensemble-size ablation), 288-tree
fractions-only median (representation ablation), 288-tree random forest mean
(bootstrap, otherwise matched tree settings), histogram gradient boosting with
160 iterations, learning rate .05, 31 leaves, minimum leaf 20, no early stopping,
squared-error loss; standardized RidgeCV on the original six alphas; fit median.
Tree counts are matched; wall time and node counts are measured, not assumed equal.
There is no independent parameter-count analogue for ridge versus trees.

For mean and median aggregation, compute calibration residuals divided by
max(tree standard deviation, .05 eV). The floor is fixed before this run. At
coverage c use sorted score rank ceil((n_cal+1)c); if rank exceeds n_cal return
an unbounded interval. Use the same calibration residuals without normalization
as a constant-width control. Negative lower limits are retained. No Gaussian
coverage or OOD guarantee is claimed. Calibration evaluates 50/80/90/95% coverage,
width, and proper interval score. This fixes the historical score-floor tuning
and small-calibration rank boundary in the successor only.

Ranking at budgets 10,25,50,100 uses (a) fraction of tree point predictions in
1.1--1.6 eV, (b) Gaussian working probability using median/tree spread,
(c) greedy distance of median prediction to 1.35 eV, and (d) 64 random rankings.
Tree fractions and Gaussian scores are ranking heuristics, not calibrated
probabilities. Evaluate only after storing score/order. Record deterministic
canonical-order tie breaking and all ranks. Comparisons are retrospective.

## Evaluation matrix and stopping

Complete 3 x 5 grouped outer cells for all models. Also evaluate Li, Zn, Ga, Se,
and Te holdouts using mean and median of the same tree ensemble with calibrated
intervals. Run first-repeat/fold-0 data-scale probes at 25% and 50% of proper
training, a +0.1 eV Gaussian training-label noise probe, and a permutation
negative control; use 96 trees for these explicitly exploratory probes. A
synthetic zero-inflated regression test probes whether median-initialized
quantile boosting has a constant first pseudo-gradient and fails to split;
compare a positive constant initialization with identical settings as a
diagnostic only, not a replacement v2 result.

Primary endpoint: paired pooled MAE difference median minus mean per repeat.
Report all baselines, folds, seeds, global/zero/nonzero/oxygen/duplicate-stratum
metrics, point RMSE, ranking, calibration and holdouts. A 5,000-resample bootstrap
resamples canonical units, averaging each unit's paired errors across repeats.
It describes fixed trained models; shared training data and adaptive project
history preclude confirmatory significance. No inferential tests are attached to
secondary endpoints; no multiplicity-adjusted superiority is claimed.

One scientific configuration; at most three hosted attempts including genuine
infrastructure repairs, one CPU hour per attempt, 512 MiB process RSS and no
rising pageouts. Do not relax guards or tune from results. Preserve failures.
Classify aggregation as SUPPORTED only if all three MAE reductions exceed .01 eV;
NEGATIVE if all differences are nonnegative; otherwise MIXED. Treat numerical
or methodological defects as INVALID and correct only affected work with a new
run ID. An interrupted run is INCONCLUSIVE, never a positive partial result.

## Reproduction and artifact contract

Run `python -m ml4materials run-robust --run-id UNUSED_ID` in the pinned hosted
environment. Save source/config/dataset hashes, versions, exact splits, checkpoints,
reload checks, row predictions, all tables, events, resource measurements and
failures. Verify complete coverage, label alignment, group isolation, and raw
metric recomputation independently of stored summary values before paper export.
Independent execution by the same operator is computational reproduction, not
independent scientific replication. External validation and synthesis remain
outside the available evidence and cannot be certified by this study.
