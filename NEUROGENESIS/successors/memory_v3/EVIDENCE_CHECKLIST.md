# Evidence checklist and unresolved limitations

This file distinguishes implemented remedies from scientific conclusions. It does not claim that all possible flaws of a research model are eliminated.

| Gate | Concrete artifact | Current evidence status |
|---|---|---|
| Canonical state and preserved history | root `RESEARCH_STATE.md`; v2 predecessor hashes; v2/v3 histories | Implemented |
| Explicit mechanism and falsifier | `SPECIFICATION.md`, `protocol.json`, `REVISION.md` | Frozen |
| Prior-art boundary | `RELATED_WORK.md`, paper bibliography | Reviewed; broad novelty not claimed |
| Full model and controls | `engine.py` | Implemented |
| Leakage, timing and checkpoints | 21 frozen successor unit tests | Passed |
| Mathematical RLS equivalence | `publication/math-verification.json` | 60 cases passed |
| First development version | v2 `runs/development-02/` | 210/210 completed; unfavorable candidate retained |
| Second/final development version | `runs/development-01/` | 210/210 completed; favorable development result |
| Preconfirmation code/config lock | `runs/freeze.json`, `runs/preconfirmation_anchor.json` | Committed and remote-ref verified before outcomes |
| Frozen full confirmation | `runs/confirmation-02/` | 1,260/1,260 complete; zero failures; SUPPORTED synthetic endpoint |
| Real temporal transfer | `runs/real-transfer-03/` | 10 attempted; 4 completed; 6 numerical failures; comparison invalid |
| Exact clean-environment replay | `verification/` or release verifier logs | All 1,260 synthetic traces/checkpoints exact; all real success/failure artifacts exact |
| Paper figures and numbers | `publication/`, `paper/` | Five figures, CSV tables and ten-page PDF compiled and visually inspected; regenerated tables/three unchanged figures match exactly |
| Kaggle execution | `HOSTED_STATUS.json`, `delivery/` | Blocked by browser transport; notebook prepared |
| Independent external replication | External party's execution/review | Not performed |
| Public/venue/arXiv submission | Submission acknowledgement | Not performed or claimed |

## Remaining scientific limitations

1. Expert reuse, delayed feedback handling and skipped updates are prior art. The study's contribution is the bounded empirical comparison, not a claimed novel general learning principle.
2. V3 changes both active updater and the absolute gate. The experiment compares the resulting mechanism; it does not isolate the separate effects of those two development changes.
3. Skipping RLS updates also changes its forgetting clock. The all-update bank is the closest registered intervention control, but it does not preserve the same downstream bank contents.
4. The main synthetic tasks remain piecewise linear. Noise, delay, dimension and horizon stress do not establish arbitrary nonlinear generalization.
5. UCI transfer is one correlated stream with imposed feedback latency. Contiguous blocks are not independent datasets.
6. An absolute loss ceiling depends on target scaling. Real transfer uses fixed physical scales and no retuning; any transfer failure must remain reported.
7. Equal memory ceilings do not imply equal parameter or operation counts. Only the measured costs support efficiency claims.
8. Full recent neural forecasting systems are reviewed but not reproduced. No state-of-the-art claim is allowed.
9. Code-path verification and same-agent clean-environment replay are not independent external replication.
10. Authorship, template redistribution rights and venue-specific requirements need owner review before an actual public submission. No venue submission is attempted by this task.

11. **Unresolved high-severity numerical flaw:** covariance-form RLS fails on the real sensor stream. A mathematically equivalent QR diagnostic still yields extreme coefficients and catastrophic baseline error. See `publication/NUMERICAL_REVIEW.md`. No stable real-world model or repaired transfer efficacy is claimed.

Final local release: `verification/release-02/receipt.json` is VALID across nine checks; 18 historical and 21 frozen successor unit tests pass; all 626 predecessor hashes are unchanged. This validates evidence integrity, not real-transfer efficacy.
