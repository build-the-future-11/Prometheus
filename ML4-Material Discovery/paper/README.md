# Complete research paper

`main.tex` uses the recovered archived `arxiv.sty` unchanged; see
`style-provenance.json`. The original project shim and draft are preserved under
`history/pre-closeout-20261021/paper/`.

Generate evidence assets from the project root:

```bash
.venv-verify/bin/python -m scripts.generate_paper_assets
tectonic paper/main.tex
```

All generated tables, figures and macros require independent numerical
verification first. The main aggregation endpoint is supported; RMSE, positive-gap
error and several shift results remain unfavorable. These are development
results, not a SOTA, independent-replication or discovery claim.

The author is Ryan Gomez, as specified by the user. No affiliation is listed.
No arXiv submission is made by this package.

The current paper is 12 pages and includes the exact fresh Kaggle reproduction
and the repaired adversarial verifier. The original 11-page package is preserved
in releases/20261021; the updated package is releases/20261021-hardening.
