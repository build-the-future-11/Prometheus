# DLML-S2-P0 frozen protocol

Frozen before any S2 task materialization.  This successor tests a limitation of S1; it does not amend S1 or turn its analytic residual into a learned controller.

## Question and falsifier

S1 only studied correctly grouped additive linear targets.  S2 asks whether the S1 analytic-residual update retains a **predeclared five-percent equal-accounted-work NMSE advantage over tuned Adam** when (a) learners receive an interleaved, incorrect two-feature partition, and (b) responses include unmodeled within-true-block interactions and quadratic additive terms.

The final-family primary claim is supported only if both conditions hold:

1. analytic residual targets improve paired equal-resource mean NMSE over Adam by at least 5%; and
2. both methods reach evaluation NMSE 0.35, with analytic residual having strictly lower median accounted work.

Failure of either condition is `NOT_SUPPORTED_ON_ADMITTED_MATRIX`.  The long-horizon family is a locked robustness/falsification check, not a tuning target.

## Information boundary

The learner receives support features, support labels, and only the supplied interleaved partition.  Generator coefficients, true partition, component values, interaction terms, and all evaluation targets remain unavailable while fitting and selecting.  Meta-train tasks reduce each candidate grid to two candidates; development tasks choose the final candidates.  Final and long-horizon outputs are never used for selection.

## Conditions

Tuned full-batch Adam, sequential analytic residual targets, simultaneous/Jacobi analytic residual targets, same-schedule block-gradient without derived targets, and fixed normalized LMS are evaluated on paired tasks.  Adam, both residual variants, and block gradient each receive a three-value grid and the same meta/development task count.  The Jacobi condition distinguishes target derivation from sequential update order.

## Families

All families have eight features, four supplied two-feature blocks, Gaussian features, noise standard deviation 0.1, and interleaved supplied groups.  Meta uses correlation 0.25, interaction 0.35, linear components.  Development uses correlation 0.45, interaction 0.55, quadratic components.  Final uses five seeds, correlation 0.6, interaction 0.8, quadratic components.  Long horizon uses five distinct seeds, correlation 0.7, interaction 0.8, quadratic components.  Support/evaluation RNG streams are independent.

## Accounting and retention

S2 retains S1's deterministic operation-proxy accounting for each existing update rule.  Checkpoints additionally retain model weights so every evaluation checkpoint can be independently read back.  Selection budget is 50,000 units, final budget 150,000, and long-horizon budget 450,000.  Every attempted candidate and every failure is written; runs terminate at the first atomic update meeting or exceeding budget, not at their best checkpoint.

## Interpretation boundary

S2 is still synthetic, CPU-only, and internally executed.  It can determine whether this exact mechanism transfers to the registered misspecified family; it cannot establish real-world utility, neural-network generalization, a learned-target result, independent reproduction, or publication readiness.
