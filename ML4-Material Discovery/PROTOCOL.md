# Frozen development protocol

Protocol version: `development-v1`  
Frozen before model execution: 2026-10-19  
Status at freeze: data artifact validated; no model result inspected.

## 1. Objective and non-objectives

Primary objective: predict experimental inorganic band gap in eV from composition alone and evaluate a calibrated policy for ranking a fixed, historical candidate pool.

Non-objectives: universal materials modeling; structure prediction; stability prediction; synthesis planning; active learning; electronic-structure calculation; wet-lab validation; prospective discovery; leaderboard submission.

## 2. Data and identifiers

- Task: Matbench v0.1 `matbench_expt_gap`.
- Rows: 4,604 in exact upstream order.
- Identifier: one-based `mb-expt-gap-{row:04d}`.
- Inputs: formula string only.
- Target: `gap expt`, eV.
- Official benchmark study: exact five outer KFold splits, shuffle true, seed 18012019.
- Custom extrapolation: each of Li, Zn, Ga, Se, and Te is fully absent from training and present in its test family; minimum test size 100.

The official rows are not edited so results remain benchmark-comparable. Canonical duplicate groups are audited. Equivalent compositions may remain across official outer folds and will be disclosed. All local calibration and custom splits are group-isolated by canonical composition. The fixed candidate pool is collapsed to one canonical composition and retains every source ID, formula, target, and target span.

## 3. Feature availability

Allowed features are 118 element fractions, number of elements, fraction entropy/min/max/L2 norm, and weighted atomic-number mean/std/min/max/range. They are invariant to element order, parentheses expansion, and formula scaling. Train element frequency state is fit on proper training data only and is used only for novelty diagnostics.

Disallowed: structure, symmetry, relaxed geometry, calculated band gap, formation energy, database identifiers, measurement metadata, or any field unavailable when a new formula is proposed.

## 4. Models

1. Trivial baseline: median target of the official outer-training rows.
2. Linear baseline: StandardScaler plus RidgeCV over alphas `[0.001, 0.01, 0.1, 1, 10, 100]`, fit only on official outer-training data.
3. Proposed predictor: ExtraTrees with 96 trees per seed, seeds 7/17/29, maximum depth 22, minimum leaf size 2, 70% features per split, bootstrap false, and one CPU worker. Predictions average all trees.

No post-result hyperparameter tuning is permitted in this protocol version.

## 5. Uncertainty and calibration

Within each outer-training fold, a deterministic label-blind SHA-256 partition assigns approximately 20% of rows to calibration using seed 731. Canonical composition groups cannot cross proper-train/calibration boundaries. Tree dispersion is floored at its calibration 5th percentile and scaled by the robust median absolute standardized residual. Normalized split-conformal intervals are evaluated at 50%, 80%, 90%, and 95% marginal coverage.

The final outer test fold is never used for fitting, model selection, dispersion scaling, or conformal scores.

## 6. Prespecified endpoints

Primary prediction endpoint: pooled official-split out-of-fold MAE (eV) for the proposed model.

Secondary prediction endpoints: RMSE, R², Spearman correlation, outer-fold metrics, three seed-specific MAEs, ridge and train-median results, formula-size/oxide/novel-element subgroup MAE, and five element-holdout results.

Primary uncertainty endpoint: observed coverage and mean width of the 90% split-conformal interval.

Abstention endpoint: MAE after retaining the 100%, 80%, 60%, 40%, and 20% lowest-uncertainty predictions, plus rejected-set MAE. A calibration curve without abstention consequences is insufficient.

Primary ranking endpoint: precision at budget 50 for an evaluation target window of 1.1–1.6 eV. Secondary budgets are 10, 25, and 100. Policies at every budget:

- random (64 fixed replicates),
- greedy predicted-property proximity to the window center,
- maximum calibrated uncertainty,
- proposed calibrated Gaussian working probability of the target interval.

The policies receive predictions and uncertainty only. True labels are supplied only to the post hoc evaluator.

## 7. Sanity and software gates

- Dataset, split, and metadata hashes must match the data contract.
- Every official identifier appears in exactly one test fold and each fold partitions all IDs.
- Local calibration and custom splits must isolate canonical composition groups.
- Missing/non-finite inputs and targets must fail closed.
- Formula order, parentheses, and integer scaling must not change features.
- eV↔meV conversions must round-trip.
- Saved checkpoints must reload with bit-identical predictions and calibrated sigma.
- Mean fold-0 MAE under three label permutations must be worse than the unpermuted proposed fold-0 MAE; failure is retained, not tuned away.

## 8. Compute admission and failures

- CPU only; `n_jobs=1`.
- Maximum observed process RSS: 512 MiB.
- Any increase in macOS pageouts after run admission is a stop condition.
- No paid APIs, simulation, DFT, cloud job, external submission, or wet-lab work.
- A run directory is immutable by convention. Failed runs remain with `FAILED_RETAINED`, traceback, events, and hashes; a repair uses a new run ID.

## 9. Interpretation

Official Matbench results are benchmark-comparable but not necessarily composition-leakage-free. Element holdouts are custom and must never be presented as official leaderboard results. A candidate ledger is a ranking record, not independent evidence of stability, novelty, synthesis, safety, or usefulness.

