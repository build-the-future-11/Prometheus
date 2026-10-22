# Decisions and history

- 2026-10-21: User authorized full implementation, bounded experiments, Kaggle execution, manuscript and corresponding GitHub delivery. The final supplied rule takes precedence over the request to continue until positive.
- S1 source and outputs preserved before changes. Original development endpoint fails: all zero/mean/ridge PCG solves use 63 iterations.
- S1 confirmatory configuration remains frozen and unconsumed. No hyperparameter or threshold edits to S1.
- New study must compare strong classical preconditioning and the direct tridiagonal solver; iteration savings alone cannot establish runtime superiority.
- Initial GitHub CLI reports invalid authentication in sandbox; verify connectivity/alternate connector before declaring permanent blocker.

- Hosted development r2 completed and was downloaded. All 26 tests pass; 3,888 S2 rows and 1,368 cases pass hashes and full semantic verification. Uploaded source matches current src/configs/tests.
- S1 tolerance audit: IID gains at rtol=0.01 are 12.864%, below the original 15% target; OOD is -4.264%. At strict tolerance both remain 0%. No S1 retuning.
- S2 development primary mean reductions: 27.757% versus zero/Laplacian and 20.740% versus physics/Laplacian. Online savings 20.362% and 26.418%. Direct solving remains approximately 4.8x faster. These are development-only observations.
- Proceed with the originally specified S2 confirmation matrix unchanged. No feature, threshold, baseline or sample-budget adaptation.
