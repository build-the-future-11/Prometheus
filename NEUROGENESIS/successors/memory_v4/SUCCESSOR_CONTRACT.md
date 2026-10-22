# Persistent-ridge memory v4: new contract before implementation/results

Previous finding: v3 improves the registered synthetic accuracy/update tradeoff but covariance RLS fails on six real methods. QR avoids the exception while revealing extreme coefficient growth. V3 release b333c415 remains immutable.

Remaining problem: forgetting removes regularization in poorly observed directions. Repeated/zero missingness indicators permit ill conditioning. This cannot be repaired by symmetrization alone.

New hypothesis: replacing decaying regularization with a fixed rho=1 information floor yields bounded conditioning under bounded inputs and preserves useful memory-based solve suppression. The exact accepted-update objective changes, so this is a successor, not a retroactive correction.

Material change: A <- lambda*A + x*x^T + (1-lambda)*rho*I, b <- lambda*b + x*y, solve A*w=b by Cholesky; A0=rho*I, b0=0. The bank/gate/information timing stay explicit. All corresponding baselines receive the same stable solver. A periodic three-label solve control tests whether merely spending fewer solves explains the benefit.

Falsifiers: any numerical failure; failure of the frozen synthetic noninferiority/update threshold; worse-than-margin new real transfer or insufficient solve savings. Real data are one sequence, so transfer thresholds are descriptive, not an IID statistical test.

Budget: one mechanism, eight gate settings, same baseline selection budget, fresh development and confirmation seeds, one locked final confirmation. New Beijing PM2.5 transfer is selected by relevance and availability before outcomes. The already exposed Air Quality stream is a regression check only. No further retuning on either final synthetic seeds or Beijing test outcomes.

Prior-art boundary: persistent regularization, forgetting corrections, and predictor reuse are known. The contribution is the explicit stable-objective/skip interaction and falsifiable bounded evaluation, not a new general-purpose learning principle. SIFt-RLS and variable-direction forgetting supply relevant competing theory; no claim of superiority over their unimplemented algorithms.

Hosted execution remains requested on Kaggle. Chrome access is currently blocked; local evidence will never be relabeled as hosted.
