# Reproducibility contract

The runnable installation, verification and reproduction commands are in
`README.md`. Exact scientific specification: `PROTOCOLS/robust-aggregation-v3.md`.

1. Read `RESEARCH_STATE.json`, install pinned hosted core dependencies, and run tests.
2. Verify pinned source data with `ml4materials prepare` on a disposable copy;
   immutable published runs must not be edited or re-manifested.
3. Run `scripts/dry_run_robust.py` on an admitted host. This is software evidence.
4. Execute `ml4materials run-robust` with an unused ID and unchanged guards.
5. Independently recompute with `scripts/verify_evidence.py`. It checks labels,
   exact partitions, all expected cells, per-model errors, interval metrics,
   ranking order, holdout outputs and resource receipts.
6. Regenerate paper assets using `python -m scripts.generate_paper_assets`.
7. Compile `paper/main.tex` with the supplied unchanged recovered `arxiv.sty`.
8. Inspect the PDF and compare generated evidence hashes before publication.

A reproduction reuses the scientific configuration; a changed model, dataset,
split rule, metric or claim requires a new successor protocol. Failed runs are
retained, not overwritten. Imported Kaggle evidence has a transport hash in
`artifacts/hosted-import-v3.json`. The executed source inventory is retained in
`artifacts/hosted-v3/freeze-v3.json`; later documentation/verifier improvements
are release-stage changes and do not rewrite that frozen inventory.

The standard-library verifier is independently coded relative to training
metrics, but it is not an independent researcher. Clean installation and one
hosted execution do not establish independently operated scientific replication.
The earlier local lock and hosted environment differ; use the recorded runtime
for training reproducibility, and treat cross-version installation as packaging
verification only. The source dataset is public and its redistribution license
and caveats are recorded in `configs/data_contract.json` and `DATA_CARD.md`.

## Verify the complete repeat

The preserved fresh execution is `kaggle-robust-aggregation-v3-reproduction-r1`.
Run `python -m scripts.compare_reproduction --output /tmp/reproduction-comparison.json`
from the full repository to verify original and repeated runs, bootstrap and
random rankings, prediction/checkpoint hashes, and scientific summaries.
`python -m scripts.verify_resampling --run-id kaggle-robust-aggregation-v3-r1`
recomputes the resampling results without importing model/statistics implementation.
The original bundle is immutable; `scripts/package_hosted.py` now refuses
to overwrite it. The reproduction launcher pins its exact SHA-256.
