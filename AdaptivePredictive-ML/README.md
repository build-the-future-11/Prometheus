# Adaptive Predictive ML

This repository contains a complete development-stage falsification study of
**Drift-Directed Fixed Share (DDFS)**, a causal online forecasting rule for
non-stationary univariate time series. DDFS combines recursive least-squares experts
with different forgetting factors, increases shared probability mass when a common
loss detector rises, and directs that mass using memory length and recent expert loss.

## Scientific outcome

Revision 2 completed on a private, CPU-only Kaggle runner. Ten registered PJM load
series were evaluated for 366 days each, producing 3,660 retained prediction rows.
The exact 23-file source payload was verified before execution and 25 hosted tests
passed.

The registered hypothesis was contradicted:

- DDFS minus fixed-share MASE: `+0.003195193449453197` (positive is worse).
- Synchronized moving-block 95% interval: `[+0.0014690581194003352,
  +0.004907738240006376]`.
- Favorable series: `0/10`.
- DDFS minus adaptive-share MASE: `+0.001023616406604363`, also `0/10` wins.
- DDFS minus AdaHedge MASE: `+0.01025341509261225`, also `0/10` wins.

Both registered advancement gates failed. Protected confirmation was therefore not
opened. The repository does not claim positive performance, generalization,
efficiency, SOTA, submission-readiness, or independent reproduction.

## Evidence map

- Canonical claim state: [`RESEARCH_STATE.json`](RESEARCH_STATE.json)
- Final execution report: [`reports/EXECUTION_STATUS.md`](reports/EXECUTION_STATUS.md)
- Frozen development protocol: [`protocols/development-v2.json`](protocols/development-v2.json)
- Raw hosted artifacts: [`artifacts/kaggle-development-v2-r1/`](artifacts/kaggle-development-v2-r1/)
- Independent recomputation receipt:
  [`artifacts/kaggle-development-v2-r1/independent-verification-v2.json`](artifacts/kaggle-development-v2-r1/independent-verification-v2.json)
- Kaggle receipt:
  [`artifacts/kaggle-development-v2-r1/remote-receipt.json`](artifacts/kaggle-development-v2-r1/remote-receipt.json)
- Complete manuscript: [`paper/main.tex`](paper/main.tex)
- Compiled manuscript: [`paper/main.pdf`](paper/main.pdf)
- Adversarial audit: [`docs/FLAW_AUDIT.md`](docs/FLAW_AUDIT.md)

## Reproduce the retained analysis and paper

The independent verifier uses only the Python standard library and imports no
research-package code:

```bash
.venv/bin/python -S scripts/verify_development_v2.py \
  --artifact-root artifacts/kaggle-development-v2-r1
```

Regenerate all result tables and PGFPlots figures from the retained JSON, then
compile the paper:

```bash
bash scripts/reproduce_paper.sh
```

For engineering development in an admitted environment:

```bash
uv sync --extra test --extra analysis
uv run pytest
uv run adaptive-predictive synthetic --output artifacts/synthetic-smoke
uv run adaptive-predictive verify --run artifacts/synthetic-smoke
```

The synthetic command is a pipeline test, not real-world performance evidence.

## Integrity boundaries

- The enclosing Atlas Git checkout contains unrelated user changes; this project
  remains uncommitted and unpushed.
- Revision-1 failures and revision-2 negative evidence are immutable historical
  records.
- Any future model change must be a separately versioned successor with a new
  hypothesis and protocol. It must not overwrite this study or use protected data to
  tune a replacement.
