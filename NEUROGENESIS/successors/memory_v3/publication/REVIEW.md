# Adversarial release review

This is a separate review pass and supplementary code path within the same project, not independent external peer review.

- Denominators: 210 v2 development attempts, 210 v3 development attempts, 1,260 frozen confirmation attempts. No synthetic failures or omitted registered cells. Ten real methods attempted; six failures explicitly retained, not removed from comparison denominators.
- Split protection: development-only selection and source/config freeze were committed and remote-ref verified before confirmation. Original v1 protected identities remain unused. QR diagnostic is explicitly post-outcome and excluded from efficacy claims.
- Leakage: cached prediction-time expert outputs, monotonic IDs, released-label runner boundary, poisoned-future-target tests, and separate timing audit. Real feature transforms use fixed scales; target missingness preserves hourly clock.
- Statistics: 20 seeds are the independent units. Five primary conditions are aggregated within each seed. A second statistical implementation reproduces the effect and interval to 1.11e-16. Contiguous real blocks are descriptive.
- Controls: selected RLS is stronger than v2 memory. All-update bank is an intervention control, but downstream bank/forgetting trajectories differ. Baseline/no-bank are identical by construction; not independent replications. Full neural prior-art systems are not executed.
- Costs: counted operations and numerical state are distinct from actual FLOPs/RSS/energy. Attempt time includes serialization. Candidate time is essentially equal to baseline despite fewer updates; no speedup claim.
- Numerical flaw found: covariance-form RLS loses positive definiteness on six real methods. Mathematically equivalent QR passes small fixtures but has catastrophic long-stream coefficient growth. Persistent regularization/rank-aware changes need a new frozen hypothesis and validation, not a hidden repair.
- Figure review found overlapping efficiency labels and arrows crossing architecture text. Generator revision fixes label positions and connects box edges. Final PDF must be rendered and checked after regeneration.
- Packaging review found a relative-path error in first claim-manifest generation. Fixed only the publication script; retained failure receipt. No frozen source or result was changed.
- Verifier review found default-only last-observation config representation mismatch. Supplementary auditor now compares normalized dataclass configs. This does not relax the frozen real efficacy verifier, which still rejects failures.
- External gates: Kaggle is prepared but not executed; independent external reproduction/review and venue submission are absent. Private GitHub upload does not imply these gates.

Conclusion: bounded synthetic claim is supported; overall scientific disposition MIXED. The model is not ready for a general real-world robustness claim. Release validation concerns artifact integrity, not successful real transfer.

- Final independent arithmetic audit: all 1,260 synthetic attempts and 1,382,400 trace rows valid; maximum streaming-sum discrepancy 2.8866e-15, within the documented 1e-12 relative / 1e-15 absolute tolerance. The prior rejection remains under verification/release-01.
