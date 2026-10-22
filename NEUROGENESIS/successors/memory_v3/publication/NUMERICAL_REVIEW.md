# Numerical failure review

The frozen real-transfer run completed four controls and retained six failed RLS-based attempts. The candidate and uniform-bank control failed after 6,702 predictions; baseline, no-bank, and all-update failed after 710; capacity-one failed after 2,425. Covariance matrices remain symmetric, but their maximum eigenvalues reach 10^15 to 10^29 while minimum eigenvalues become negative. Symmetrization is therefore not a sufficient repair.

The decaying ridge and correlated/missingness indicator features permit extreme conditioning. The exact mathematical recursion passes small-system checks, while the covariance-form floating-point implementation does not remain numerically reliable on this stream.

A separate, explicitly post-outcome square-root information QR diagnostic is proposed in qr_diagnostic.py. It changes the numerical representation while retaining the same accepted-update ridge objective, with no new hyperparameters. It is not substituted into the frozen candidate or treated as a new protected success. Full qualification would require versioned checkpoints, source locks, matched reruns, and a new unexposed external validation dataset. No third mechanism search is authorized by the registered budget.

## Executed diagnostic

The QR formulation matched the batch weighted-ridge solution in 30 small-system cases (maximum coefficient error 5.55e-16) and reached all 9,357 real rows. It did not provide an acceptable repair: the ordinary RLS control produced test MSE about 2.45e62 and maximum coefficient magnitude about 1.85e47; the candidate also retained coefficients above 1e8. Removing the immediate covariance exception therefore does not establish numerical robustness. The diagnostic is retained in qr-diagnostic.json and is excluded from all main efficacy tables.

A durable repair likely requires a persistent ridge, rank-aware treatment of redundant features, or a revised forgetting policy. These change the statistical model/objective and need a versioned hypothesis, predeclared controls, and new unexposed validation. They are not silently patched into v3 after protected-data exposure. This is a concrete unresolved high-severity flaw, not an infrastructure blocker or a favorable model result.
