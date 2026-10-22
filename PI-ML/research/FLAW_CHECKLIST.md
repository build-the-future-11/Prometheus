# PI-ML adversarial flaw and repair register

Severity: P0 invalidates a claim; P1 blocks completion; P2 limits scope. Status is evidence-dependent.

| ID | Flaw and cause | Severity | Repair | Required acceptance evidence |
|---|---|---|---|---|
| F01 | S1 learned and zero starts require 63 iterations at 1e-10; solution accuracy is not Krylov acceleration | P0 | Preserve negative result; inspect tolerance dependence and preconditioning separately | Retained S1 raw cases, tolerance matrix; no retroactive endpoint change |
| F02 | Direct Thomas solver solves the tridiagonal problem in linear time | P0 | Include direct solve in every successor cell; forbid a practical superiority claim unless beaten | Full online timing includes predictor, assembly, residual certification |
| F03 | Jacobi alone is an incomplete classical baseline set | P0 | Add constant-Laplacian inverse preconditioner and physics-only start | Paired same-input same-tolerance results |
| F04 | Warm starts and POD/learned Krylov ideas have extensive prior art | P0 | Position as controlled empirical study, no new solver-family claim | Primary-source literature matrix, explicit novelty boundary |
| F05 | Quadratic ridge mixes forcing terms that violate exact source linearity | P1 | Versioned source-linear coefficient-polynomial predictor; compare unconstrained ridge | Superposition/zero-forcing tests, identical train cases, fresh test split |
| F06 | S1 ridge refactors the same Gram matrix for every output | P2 | Successor solves multiple RHS with one factorization | Numerical comparison with S1 predictor on hand-sized data |
| F07 | Recursive residual alone can miscertify floating-point convergence | P0 | Recompute true residual, restart when necessary, retain both histories | Independent residual check for every converged solve and numerical unit tests |
| F08 | S1 conformal scores pool correlated grid points and interpolate quantiles | P0 | Withdraw distribution-free coverage claim; do not carry uncertainty intervals into S2 | Explicit manuscript limitation; no coverage guarantee asserted |
| F09 | S1 timing uses fixed order, one sample and no complete offline accounting | P1 | Randomized method order, repeated timing, data-generation and fit times reported | Raw repeated times, median per case, environment and CPU/RSS receipt |
| F10 | Reusing train seeds would overstate model generalization | P1 | Three independent development and five independent confirmatory training sets; paired cases within each block | Disjoint seed ledger, block-aware bootstrap and all training replicates shown |
| F11 | Multiple endpoints could enable cherry-picking | P0 | Single prospective primary cell and joint baseline rule; all others descriptive | Frozen config/source digests before protected evaluation |
| F12 | Existing verifier checks hashes but not regenerated science | P0 | Successor replays cases/predictions/residuals and recomputes analysis from raw rows | Corruption tests, artifact identity checks and semantic verification receipt |
| F13 | No canonical state or change history | P1 | RESEARCH_STATE plus append-only decisions and preserved predecessor | State agrees with source, outputs and remote receipts |
| F14 | No manuscript, README, release archive or source receipt | P1 | Complete documents, source manifest, figures, archive and reproducibility commands | Build, paper compile/render, clean-copy replay and hashes |
| F15 | Local host memory pressure and restricted process inventory | P1 | Record resource admission; use bounded CPU Kaggle execution | Actual hosted completion and downloaded output verification |
| F16 | Synthetic 1D task is not evidence for realistic multidimensional performance | P0 | Restrict all conclusions; include resolution, coefficient and forcing shifts | Full matrix; explicit absence of 2D/real-data/SOTA claims |
| F17 | Benchmark reference and implementation share code | P1 | Independent dense solve check in tests; manufactured-solution accuracy test | Independent reference agreement and refinement trend |
| F18 | Baseline and proposed model use different feature counts | P1 | Report counts and use equal-count shuffled-label control; no parameter-isolated causal claim | Capacity table and shuffled control; mechanism claim qualified |
| F19 | Code can drift after outcomes are seen | P0 | Source snapshot in each run plus frozen source hash | Verifier rejects mismatching source; successor for material changes |
| F20 | A same-host rerun is not independent research reproduction | P1 | Name it clean-environment replay; preserve external review as unfinished when unavailable | Separate reproducibility and independent-verification statuses |
