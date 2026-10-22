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

- Frozen source commit 66d5721, freeze commit e864d62. Remote branch codex/pi-ml-s2-study independently read back at e864d62ef365bbf3ae5e2f0e2d464437bf96ce61.
- S1 protected confirmation: NEGATIVE, exact 0% IID/OOD iteration reduction, 128 cases per test split. This outcome is terminal and preserved.
- S2 protected confirmation: SUPPORTED for the predeclared IID/129/1e-6 joint iteration claim. 27.725% reduction vs zero/Laplacian, 21.172% vs physics/Laplacian; time reductions 21.720% and 26.071%. All 17,280 rows converge. No post-outcome scientific code or protocol edits.
- Producer uses 175.592 CPU seconds, 175.753 wall seconds, 191.234 MiB peak RSS before verifier. Full hosted confirmation wrapper 257.678 wall seconds. Direct solver mean 0.4485 ms versus proposed 2.5180 ms: no overall fastest-solver claim.
- Independent implementation audit and paper rendering are verification work only; no new hypothesis search or protected rerun.

- Release-audit r1 failed before tests at Kaggle venv ensurepip bootstrap. No scientific rerun or source change. r2 creates a venv without ensurepip and installs pip from the official PyPA bootstrap, retaining its hash.

## Final artifact audit
Release-audit r2 completed: 26 installed-wheel tests, full solver/analysis replay, separate SciPy/flux/exact-bootstrap checks and manuscript build passed. Its downloaded files match the archive manifest. Final editorial rendering is a separate manuscript-only job; it changes no scientific code, protocol, inputs or outcomes. API pagination through the retained clean environment was slow; 100-file metadata pages and a matching browser download recovered the 650,609-byte audit archive.

## Publication closure
Full release fe85196 published and remote-ref identity verified; PR55 created. GitHub Actions was refused before any job step by account billing/spending eligibility. Record this external gate; do not bypass checks, change billing or claim GitHub CI passed. Kaggle verification remains the executed engineering and numerical evidence. The defined study is complete; new scientific hypotheses require a successor.
