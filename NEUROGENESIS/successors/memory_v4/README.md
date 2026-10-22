# NEUROGENESIS v4: persistent-ridge successor

Status: **SYNTHETIC_CONFIRMED_REAL_PENDING**. This is implemented research code with a verified synthetic confirmation, not yet a complete real-transfer study. The completed v3 paper and outcomes remain unchanged in ../memory_v3.

Read SUCCESSOR_CONTRACT.md, SPECIFICATION.md, protocol.json, FLAW_CHECKLIST.md and STATUS.json before continuing. The active estimator now maintains a fixed ridge information floor and solves by Cholesky. A periodic solve control tests the simpler explanation for update savings. The gate and finite snapshot bank retain the v3 mechanism. Persistent regularization changes the objective; it does not retroactively repair v3.

26 tests passed, including 1,200 independent batch comparisons, 5,000 steps with unexcited/duplicate features, periodic sufficient-statistic accounting, delayed-feedback legality and exact checkpoint restoration. These are engineering and mathematical checks, not evidence of better prediction.

Development-01 was refused before any trial: pageouts rose from 137945 to 137947. A separate later probe rose from 140061 to 140125. Those receipts remain preserved. Fresh development-02 passed admission and completed all 220 registered attempts with zero failures. Exact replay verified 211,200 trace rows and the separate auditor returned VALID. Selection froze `bank_c0.1_a0.5_k8` against `rls_0.95` in `runs/freeze.json`. No protected model outcome has been evaluated and v4 has not been uploaded to GitHub.

The freeze is anchored locally at commit `dd71324e6e262ebd60bad5b57a7b3071802a7af5`, directly parented to published v3 commit `b333c415ca9a50aa468b48ef7675b6984228e709`. Confirmation-01 then refused admission before any model attempt because pageouts rose from 222881 to 222882; a follow-up two-second probe rose from 222915 to 222919. The guard remains binding. Resume under a fresh confirmation run ID only after a stable interval.

After resources stabilized, confirmation-02 completed all 1,400 registered attempts with zero failures. Exact replay verified all 1,536,000 trace rows and checkpoints; the separate auditor also returned VALID. The primary paired result is a 30.6247% MSE reduction against the selected baseline (95% bootstrap interval 27.4379%--33.9694% reduction) with 58.8905% fewer weight solves. The pooled novel-regime condition is 0.2856% worse than baseline. This is a bounded synthetic result, not evidence of real-stream robustness or state of the art.

## Resume after resource recovery

1. Run `transfer.py run --dataset air` for exposed regression and `--dataset beijing` for the new protected stream, each with `--lock runs/freeze.json --output` a fresh directory and a passing resource admission.
2. Run exact replay, the separate arithmetic audit, evidence-derived figures and the complete successor manuscript. Keep negative results and failed cells.
3. Execute the frozen package on Kaggle when an authenticated working connection is available; preserve its actual output archive. A Google Chrome or Kaggle tab alone is not execution evidence.
4. Scan and publish only this project in the corresponding private Atlas repository; verify remote SHA/tree. Do not alter the closed v3 subtree or unrelated parent work.

New data: Chen, S. (2015), Beijing PM2.5, UCI Machine Learning Repository, doi:10.24432/C5JS49, CC BY 4.0. Official source and exact file hashes are in data/manifest.json. 2013 is warmup; 2014 is evaluation. Air Quality is already exposed and cannot be called fresh validation.
