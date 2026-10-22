# DLML-S1-P0 frozen protocol

Frozen before the admitted development run on 2026-10-19. Changing the hypothesis, split seeds, endpoint, threshold, accounting formulas, or candidate grid requires a versioned successor protocol rather than editing an outcome.

## Question and falsifier

Primary hypothesis: analytically derived block-residual targets improve both (1) mean held-out NMSE at equal accounted arithmetic work and (2) work required to reach NMSE 0.08 versus tuned Adam on the admitted final family.

The hypothesis is supported on this matrix only if:

1. relative improvement in paired equal-resource mean NMSE over Adam is at least 5%; and
2. both methods reach NMSE 0.08, with a strictly smaller median work-unit count for the proposed method.

Failure of either clause yields `NOT_SUPPORTED_ON_ADMITTED_MATRIX`. Long-horizon results are a robustness check, not a replacement endpoint.

## Information boundaries

Each task has independent support and evaluation RNG streams. Training receives only support features/labels and the public feature-group partition. Coefficient vectors and component targets remain hidden. Meta-training tasks screen candidates; development tasks choose among the two best meta-training candidates; final and long-horizon answers never affect selection. Evaluation labels score frozen checkpoints only.

## Declared matrix

- Meta-training: seeds 101–103, correlation 0.1, coefficient scale 1.0.
- Development: seeds 201–203, correlation 0.3, coefficient scale 1.2.
- Final unseen family: seeds 301–305, correlation 0.6, coefficient scale 1.4.
- Longer-horizon unseen family: seeds 401–403, correlation 0.7, coefficient scale 1.4, triple final work budget.
- Every family uses eight features, four disjoint two-feature blocks, Gaussian inputs, additive linear targets, and noise standard deviation 0.1.

All methods receive the same task instances. Candidate grids and budgets are machine-readable in `configs/development.json`.

## Conditions

1. **Tuned Adam:** conventional full-batch MSE optimization; learning rate selected from 0.01, 0.03, 0.08.
2. **Analytic residual targets (primary):** sequential block ridge solves; regularization selected from 0.01, 0.1, 1.0.
3. **Fixed normalized LMS:** fixed step 0.25, no selection.
4. **No-derivation block-gradient ablation:** same sequential block schedule but ordinary MSE gradients; learning rate selected from 0.005, 0.02, 0.08.

The equal-resource analysis uses, per paired task, the smallest terminal accounted budget across all four conditions and the last checkpoint at or below it. The block-gradient condition is the same-budget ablation at the granularity of a complete predeclared block sweep; any atomic-step overshoot remains visible in the terminal-work column.

## Resource accounting

Arithmetic work units are deterministic algorithmic operation proxies, not wall-clock claims. The implementation records support-example reads, teacher calls, target-generator training, probes, preprocessing, and global-gradient calls separately. Task generation and evaluation are common across conditions and excluded from optimization work; all non-common learning work is included. Single-process CPU only, no remote compute, paid services, or worker fan-out.

Adam step: `4*n*d + 12*d`. Analytic sweep: prediction `2*n*d`, plus for each block of width `b`, `n*(2*b^2 + 6*b + 4) + 2*b^3`. Fixed LMS step: `5*n*d + 4*d`. Block-gradient sweep sums `2*n*d + 2*n*b + 3*b` per block.

## Failure retention and stopping

Every candidate/task attempt is written to JSONL. Exceptions and non-finite metrics become retained `FAILED` records. The endpoint is the first atomic update whose cumulative work meets or exceeds the declared budget; results are not stopped at the best observed checkpoint. No rescue tuning is allowed after final results are read.
