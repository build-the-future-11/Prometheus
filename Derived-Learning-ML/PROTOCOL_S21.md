# DLML-S2.1-P0 frozen protocol

S2.1 is a new successor frozen after S2 exposed that its nonlinear target family had nonzero mean while the learner lacked an intercept.  S2 remains retained, unchanged, and is not reinterpreted.  S2.1 supplies every method an affine intercept and repeats the structural-misspecification question on distinct seeds.

## Registered question

On the same wrong-partition, interaction-plus-quadratic family class as S2, does sequential analytic residual updating achieve at least a five-percent paired equal-work NMSE improvement over development-selected Adam, and lower median work to evaluation NMSE 0.35, when both models can learn an intercept?

Both clauses are required for `SUPPORTED_ON_ADMITTED_MATRIX`; otherwise the result is `NOT_SUPPORTED_ON_ADMITTED_MATRIX`.  Final and long-horizon labels remain unavailable during selection.  Long horizon is a robustness/falsification split.

## Matrix and controls

The five conditions, three-value candidate grids, `meta_top_k=2`, task sizes, structural parameters, and work budgets are machine-readable in `configs/s21_structural_misspecification_intercept.json`.  Meta, development, final, and long-horizon seeds are all distinct from S2.  Each task exposes only support data and an interleaved supplied partition; coefficients, true partition, components, interactions, and evaluation targets are sealed while fitting.

## New affine capacity

Every condition uses the same scalar intercept.  Analytic residual uses its support-loss minimizing intercept update; Adam and gradient/LMS rules update the intercept under their own declared update schemes.  Intercept state and every checkpoint intercept are retained and independently read back.  This adds a necessary common model capability; it does not make the residual target learned or correct the wrong partition/non-additive mismatch.

## Scope

S2.1 is a CPU-only internal synthetic study.  It can test this registered mechanism boundary, including a negative outcome.  It cannot substantiate real-data utility, neural scaling, learned-controller claims, independent reproduction, or submission readiness.
