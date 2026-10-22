## Verified follow-up: exact reproduction and release 0.1.1

The fresh run `kaggle-robust-aggregation-v3-reproduction-r1` used the exact
original frozen source and settings. All 20 checkpoints, prediction files,
splits, units and diagnostic outputs match byte for byte. Scientific summaries
match exactly after excluding timing fields. Runtime was 369.8 seconds at
372.1 MiB peak RSS with zero pageout increase. This is same-operator computational
reproduction, not independent scientific replication or protected confirmation.

The repaired verifier rejects four rehashed corrupted copies accepted by its
predecessor. Release tests: 72 passed. Independent arithmetic recomputed the
5,000-resample confidence interval and 12 random-ranking summaries exactly.
Report generation now preserves publication/authorship state, and packaging
refuses to overwrite the original freeze. Source distributions exclude prior
release archives. See `artifacts/reproduction-comparison-20261021.json`,
`artifacts/adversarial-audit-20261021.json`, and `FLAW_AUDIT.md` (38 findings;
the newest is a literature-boundary repair and does not alter frozen results).

The positive primary endpoint and adverse RMSE, positive-gap and chemical-shift
results are unchanged. Independent external validation and venue-specific
submission remain open. Ryan Gomez is the paper author; no affiliation is listed.
