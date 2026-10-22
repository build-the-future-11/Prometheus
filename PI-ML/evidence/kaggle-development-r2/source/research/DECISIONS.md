# Decisions and history

- 2026-10-21: User authorized full implementation, bounded experiments, Kaggle execution, manuscript and corresponding GitHub delivery. The final supplied rule takes precedence over the request to continue until positive.
- S1 source and outputs preserved before changes. Original development endpoint fails: all zero/mean/ridge PCG solves use 63 iterations.
- S1 confirmatory configuration remains frozen and unconsumed. No hyperparameter or threshold edits to S1.
- New study must compare strong classical preconditioning and the direct tridiagonal solver; iteration savings alone cannot establish runtime superiority.
- Initial GitHub CLI reports invalid authentication in sandbox; verify connectivity/alternate connector before declaring permanent blocker.
