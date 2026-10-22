# NEUROGENESIS: results, status, and next actions

Date: 2026-10-21. Overall scientific disposition: **MIXED**.

## Results across iterations

| Version / phase | What actually ran | Finding |
|---|---|---|
| Historical v1 | 20 baseline development attempts; no candidate | A benchmark exists, but no candidate efficacy claim. Protected v1 identities remain untouched. |
| Memory v2 development | 210 attempts; five seeds, two conditions | Best memory candidate MSE 0.80245 versus RLS 0.33732: unfavorable. Preserved without deletion. |
| Memory v3 development | Same 210-attempt development structure | Selected memory candidate MSE 0.26350 versus RLS 0.33732. This was selection evidence only. |
| Frozen v3 confirmation | 1,260 attempts; 20 fresh seeds, seven conditions, nine methods | Primary paired relative MSE 36.01% lower (95% bootstrap interval 34.00%-38.04% lower); 58.55% fewer weight updates. Frozen hypothesis SUPPORTED. |
| Real temporal transfer | Ten methods on UCI Air Quality, fixed synthetic-selected settings | Four controls completed; six RLS methods, including candidate and baseline, failed numerically. Real-transfer efficacy not established. |
| Post-outcome QR diagnostic | 30 math cases plus two real-stream diagnostic runs | Immediate exceptions avoided, but extreme coefficient growth and catastrophic baseline error remain. Not an accepted repair or a protected efficacy result. |

Development and confirmation have different seeds/condition sets; their MSE values are not a single learning curve. The exact same synthetic confirmation traces and checkpoints reproduced in a clean virtual environment (1,382,400 trace rows). The same real successes/failures also reproduced exactly (55,387 retained trace rows).

## What is supported

The eight-expert delayed-credit RLS memory gate improves the registered synthetic accuracy/update tradeoff against the development-selected RLS control. The paired candidate change against the always-update bank is -22.66% (95% interval -24.65% to -20.73%), descriptive mechanism evidence. In the no-recurrence condition its pooled MSE is 1.48% worse than RLS. The primary claim is bounded to the declared linear generators, not universal generalization.

Fewer updates are not a hardware speed/energy claim. The generated table reports numerical state and counted arithmetic separately. The baseline and no-bank rows coincide because the selected baseline is RLS with the same forgetting factor.

## What failed and why

Covariance-form RLS becomes ill-conditioned on the long correlated sensor stream. Baseline/no-bank/all-update fail after 710 predictions; capacity-one after 2,425; candidate/uniform after 6,702. The full test interval is not available for those methods, so no partial-score comparison is admitted. The covariance remains symmetric but loses positive definiteness. A decaying ridge leaves weakly observed/redundant directions unstable. A square-root QR diagnostic does not resolve the underlying robustness issue.

The four completed real controls are FIFO replay (MSE 0.551697), FIFO kNN (0.913742), last observation (3.228987), and frozen zero (7.606414), in original CO units squared, each over 4,668 test targets. These values do not support a candidate transfer claim.

## Current status

- Implemented: candidate, matched controls, timing boundary, checkpoints, tests, raw evidence, frozen selection, statistical analysis, mathematical checks, independent arithmetic code path, exact same-implementation replay.
- Science: synthetic hypothesis supported; real comparison invalidated by numerical failure; overall MIXED. The bounded two-version search is closed, not continued until all results are positive.
- Paper/release: see final paper PDF, publication claim ledger, release verification receipt, and remote publication receipt.
- Hosted: prepared, **not executed**. A Kaggle draft exists, but browser transport failed and no local API credentials are configured. No Kaggle result is claimed.
- External replication/review: not performed. Clean-environment replay by this project is not independent external replication.
- Public venue/arXiv submission: not performed. Authorship/template/venue requirements remain owner-review items before a public submission.

## What you can do now

1. Read the paper and use the implementation/checkpoints as a research prototype within the tested synthetic scope.
2. Restore the browser extension connection or configure Kaggle API credentials locally, then import `successors/memory_v3/delivery/neurogenesis-memory-v3-evidence-r3.ipynb` into a CPU notebook, run it, and retain the output archive. Do not paste credentials into chat.
3. For a robust model successor, first freeze a persistent-regularization or rank-aware hypothesis, matched controls, and new unexposed real validation. The existing real stream is now diagnostic/development evidence for that successor; do not relabel it as fresh validation.
4. Obtain independent review/reproduction before external scientific or deployment claims.

## Completion assessment

The original request is **not fully complete**: actual Kaggle execution is blocked, independent external review is absent, and the model has an unresolved numerical robustness flaw. Local bounded experiments and their honest scientific conclusion can be complete even when the model is not a deployable solution. A percentage would hide these distinct gates.

## Evidence pointers

- `successors/memory_v3/runs/freeze.json` and `preconfirmation_anchor.json`: source/configuration lock before outcome review.
- `successors/memory_v3/runs/analysis.json`: primary statistics and every condition.
- `successors/memory_v3/verification/clean-replay-01/confirmation.json`: exact full confirmation replay.
- `successors/memory_v3/verification/real-outcome-01/replay.json`: exact successes and numerical failures.
- `successors/memory_v3/publication/statistics-verification.json`: independently implemented bootstrap arithmetic.
- `successors/memory_v3/publication/math-verification.json`: 60 RLS/batch-ridge equivalence cases.
- `successors/memory_v3/publication/NUMERICAL_REVIEW.md`: failure diagnosis and rejected QR repair diagnostic.
- `successors/memory_v3/HOSTED_STATUS.json`: precise hosted blocker.
