# PI-ML: source-linear warm starts for elliptic solvers

An executable, evidence-bound synthetic 1D elliptic solver study. Read `RESEARCH_STATE.json` first. The original S1 strict-tolerance development result is preserved: zero iteration savings. S2 studies a separately specified source-linear predictor with strong classical controls. Confirmation results are admitted only from the frozen, verified run; package/test success alone is not scientific support.

## Verified results

Private Kaggle confirmation: **S2 primary SUPPORTED**, five independent training blocks and 160 primary cases. At N=129 and relative residual tolerance 1e-6:

| Method | Mean iterations | Mean online time |
|---|---:|---:|
| Zero + Laplacian PCG | 12.556 | 3.212 ms |
| Physics start + Laplacian PCG | 11.556 | 3.393 ms |
| Source-linear start + Laplacian PCG | 9.131 | 2.518 ms |
| Direct Thomas solve | 0 | 0.448 ms |

Mean paired iteration reductions are 27.73% versus zero and 21.17% versus physics; mean paired online-time savings are 21.72% and 26.07%. Direct remains 5.61x faster by the ratio of mean times. All 17,280 full-matrix S2 rows converge, but forcing-OOD iteration gains against physics disappear. **S1 original confirmation remains NEGATIVE at 0%**.

See the [full paper](release/PI-ML-study.pdf), [release ledger](research/RELEASE.md), raw metrics, confidence intervals, all controls and limitations. These are actual hosted measurements on a declared synthetic PDE family, not real observational data.

## Install and use

Python >=3.11. The hosted frozen environment is Python 3.13.15 / NumPy 2.1.3.

```sh
python -m pip install '.[research]'
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
python -m piml verify evidence/s1/development-S1-v1
python -m piml.s2 run configs/s2_development.json runs/new-development
python -m piml.s2 verify runs/new-development
```

The original confirmatory outcome is consumed once and retained. An explicitly labeled replication can use a new output directory and must match the frozen code/config:

```sh
python -m piml.s2 run configs/s2_confirmatory.json runs/new-replication --freeze research/S2_FREEZE.json
```

This command never overwrites old results. A different scientific model or protocol requires a successor version. Read the resource budget in `RESEARCH_STATE.md` before launching numerical work.

## Structure

- `src/piml/`: complete S1 and S2 implementations, statistics, runners and verification.
- `configs/`: original configurations, smoke/development protocols and frozen confirmation.
- `tests/`: mathematical, numerical, leakage, artifact and freeze regressions.
- `research/S2_SPECIFICATION.md`: exact model, assumptions, costs, controls and prospective statistical rule.
- `research/FLAW_CHECKLIST.md`: flaw, cause, severity, repair and acceptance evidence.
- `research/DECISIONS.md`: retained history and decisions.
- `evidence/s1/`: byte-preserved predecessor source/configs/tests and development run.
- `evidence/kaggle-development-r2/`: downloaded private hosted development source and measurements.
- `paper/`: full LaTeX manuscript using the user's unchanged arxiv.sty, source-derived tables and figures.
- `scripts/package_kaggle.py`: secret-free private CPU source packaging.
- `scripts/independent_audit.py`: separately implemented SciPy/flux/finite-bootstrap checks.
- `scripts/build_paper.py`: regenerate numerical prose, tables, figures and claim links from retained evidence.

## Scope

The task is synthetic scalar diffusion on a 1D interval, not real observational data. The model is polynomial regression, not a deep network. Positive iterative results cannot establish practical superiority over the faster direct tridiagonal baseline or generalize to 2D/3D automatically. All failed endpoints, OOD results and costs are retained. Independent implementation checks are distinguished from reproduction by an independent investigator.

## Paper and artifact reproduction

After installing the research extra, regenerate the paper inputs with the two downloaded results directories and compile `paper/main.tex` using Tectonic or a standard LaTeX installation. The exact commands, checksums and release disposition are in `research/RELEASE.md`. The private GitHub location is `build-the-future-11/Atlas`, subdirectory `PI-ML`.
