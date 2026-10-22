# Quantile-composition-v2 protocol

Status: **IMPLEMENTED, NOT EXECUTED**

This successor is separate from frozen `development-v1`. It evaluates direct conditional quantile gradient boosting for composition-only experimental band-gap prediction on the same pinned Matbench v0.1 task and exact official outer-fold IDs. It does not alter the v1 data, candidate ranking, ExtraTrees ensemble, resource receipt, or failed-run artifacts.

The point model uses gradient boosting with absolute-error loss. For each central interval coverage in `configs/studies/quantile-composition-v2.json`, two gradient-boosting models estimate the lower and upper conditional quantiles. Models fit only each official outer-training fold. The outer-test fold is used only for final MAE, empirical interval coverage, mean width, and the proper interval score. Results remain benchmark-comparable but inherit the documented canonical-composition overlap caveat in the official Matbench splits.

The primary endpoint is pooled official-split out-of-fold MAE. The uncertainty endpoint is observed 90% coverage and mean interval score. Direct quantile intervals must be compared with the frozen ExtraTrees plus normalized split-conformal intervals only on the exact same outer-test rows; neither approach is prespecified to win.

Every run uses a new ID, the unchanged 512 MiB maximum RSS and no-rising-pageout admission condition, immutable receipts, prediction tables, checkpoint hashes, and artifact manifests. No execution result exists when this protocol is created. A model checkpoint, fixture test, or incomplete fold is engineering evidence only.
