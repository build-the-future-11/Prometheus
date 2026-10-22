# Literature boundary (checked 2026-10-21)

This is a targeted primary-source review, not an exhaustive novelty certificate.

| Source | Relevant precedent | Consequence for PI-ML |
|---|---|---|
| Zhang et al., Blending Neural Operators and Relaxation Methods in PDE Numerical Solvers (2024 revision), https://arxiv.org/abs/2208.13273 | HINTS combines learned and classical correction with spectral complementarity | Hybrid correction is not new; no HINTS reproduction or superiority claimed |
| Eshaghi et al., NOWS (2026 revision), https://arxiv.org/abs/2511.02481 | Learned initial guesses for Krylov solvers | The warm-start concept is established |
| Carlberg, Forstall, Tuminaro (2016), https://arxiv.org/abs/1512.05820 | POD-augmented CG and recycling | A learned coarse space alone would not establish novelty |
| Kahl and Rittich (2017 revision), https://arxiv.org/abs/1209.1963 | Deflation subspace accuracy and effective conditioning | Better solution norm alone does not imply convergence improvement |
| Luo et al., Neural Krylov Iteration (NeurIPS 2024), https://proceedings.neurips.cc/paper_files/paper/2024/file/e88870ec82f2469b0ddf32c817920c68-Paper-Conference.pdf | Learned Krylov acceleration | Must compare both learning and numerical mechanisms |
| Transferability of Learned States in Neural PDE Solvers (2026), https://arxiv.org/abs/2610.10972 | Distinguishes correctness, learning contribution and complete numerical utility | Directly overlaps evaluation concerns; our contribution is a small reproducible audit, not a new conceptual framework |

## Defensible contribution
An auditable controlled study of a failed quadratic warm start and a source-linear successor on a fully specified synthetic 1D family, separating initial-guess quality, preconditioner effects, strict true-residual acceptance and end-to-end cost. No new state-of-the-art algorithm claim is warranted by this scope.

## Hypothesis and falsifier
For coefficient-varying elliptic systems, preserving exact linearity in forcing may improve starting fields; any practical claim is falsified when the gain disappears against a Laplacian-preconditioned zero/physics start or the direct tridiagonal solve after full online costs. A failure is retained as a result.
