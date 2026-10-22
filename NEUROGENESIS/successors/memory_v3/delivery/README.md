# Frozen Kaggle execution

Import `neurogenesis-memory-v3-evidence-r3.ipynb` through Kaggle's notebook File menu and use a CPU session. The notebook embeds the complete frozen source, the public UCI dataset, its provenance manifest, and the locked configuration. No internet or pip install is required on Kaggle.

It executes correctness tests, the full 1,260-attempt matrix, separate verification, fixed analysis, and all ten real-transfer methods. The configuration is not tuned on hosted outcomes. The expected output is `/kaggle/working/neurogenesis-memory-v3-results.zip`; download it only after the notebook reaches `FINAL` and an EVIDENCE_RETAINED terminal record exists.

The archive contains `ARTIFACTS.json`, raw traces, checkpoints, all source files, run manifests, verification receipts, and `terminal.json`. Verify every archive hash, compare source digests against `runs/freeze.json`, and compare scientific results with the retained local run. Runtime and platform metadata may differ. A matching replay establishes cross-environment reproducibility using this implementation; it does not establish independent external replication.

Current upload/execution state is in `../HOSTED_STATUS.json`. This directory alone is preparation, not evidence that Kaggle ran the study.

Delivery revision 2 preserves the frozen source archive and adds a supplementary failure-aware auditor. The strict original real-transfer efficacy verifier still rejects failed cells. Known local outcome: six RLS-based methods fail numerically; no candidate real-transfer MSE or improvement is admitted. All failure artifacts remain in the downloaded archive. The earlier notebook is retained as delivery history.

Delivery revision 3 also includes the supplemental rounding-aware arithmetic auditor; frozen `audit.py` is retained as provenance. Original exact floating-point equality rejects some valid means by about 3e-15. Model source, source archive checksum, configuration, selection, and efficacy thresholds remain unchanged.
