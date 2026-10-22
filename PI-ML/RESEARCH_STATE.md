# PI-ML canonical research state

Read RESEARCH_STATE.json, research/DECISIONS.md, research/CHECKPOINTS.md and research/RELEASE.md before work. Update state before finishing.

## Scientific disposition
- S1: original protected confirmation NEGATIVE; 0% iteration reduction on 128 IID and 128 coefficient-OOD cases. Historical source and development evidence remain byte-preserved in evidence/s1.
- S2: SUPPORTED for the fixed IID, N=129, rtol=1e-6 joint iteration hypothesis against both zero and physics Laplacian starts. Five training blocks, 160 primary cases, 17,280 full-matrix rows, zero convergence failures.
- Primary mean paired reductions: 27.725% and 21.172%; corresponding online-time reductions: 21.720% and 26.071%.
- Direct tridiagonal solving remains faster (0.4485 ms versus 2.5180 ms mean). Forcing-OOD iteration reductions against the physics baseline are negative. Do not claim universal improvement, a fastest solver, a novel solver family or real observational data.

## Current delivery stage
Complete research release `fe851967cc1287da3f0e5e7603904151798b8b0d` is saved on private Atlas branch `codex/pi-ml-s2-study`; remote ref verified. PR: https://github.com/build-the-future-11/Atlas/pull/55. A following documentation commit records the publication receipts.

The 12-page paper is compiled and visually inspected. All 26 installed-wheel tests, complete solver/analysis replay and separate SciPy/flux/exact-bootstrap audit passed on Kaggle. Code, raw results, failed attempts, arXiv source package and research supplement are complete and hashed.

GitHub Actions is BLOCKED BEFORE EXECUTION by account billing/spending eligibility; no GitHub CI success is claimed. Restore account eligibility and rerun checks before merge. The branch is published; main is not merged. Independent investigator reproduction and venue submission remain external steps.

## Budget and protection
At most two specified candidate degrees; development and confirmation are consumed and must not be rerun for better outcomes. Keep 512 MiB free-plus-speculative reserve and stable pageouts for heavy local work. No GPU or paid compute. Hosted scientific producer CPU so far: 198.302 seconds for S2; full wrapper and verification costs are separate receipts. Original bound: 3600 hosted CPU seconds plus bounded verification, 600 local CPU seconds.

## Repository and scope
Canonical remote: build-the-future-11/Atlas, subdirectory PI-ML. Publication uses an isolated clone to preserve unrelated dirty parent files. Model is source-linear polynomial regression on synthetic 1D PDEs. Independent implementation checks are distinct from external independent investigator reproduction, which remains unperformed.
