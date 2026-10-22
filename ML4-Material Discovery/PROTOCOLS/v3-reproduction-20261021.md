# Exact v3 computational reproduction verification

Freeze before execution: 21 October 2026. The purpose is to test reproducibility of the existing result, not select a better score. One full run is allowed, plus one retry only for an evidenced infrastructure failure. No model, split, metric, hypothesis, source transport or dataset changes are allowed. Maximum one CPU hour per attempt; 512 MiB RSS; no increasing pageouts; no paid compute.

Use the original frozen source bundle with SHA-256 b54be749fc45aaa81625a5fea82e61c83f968db9e9798ee5e49416e6bd0787fb. The new launcher changes the run ID/work directory and explicitly pins this transport hash. It executes the original tests, synthetic dry run, all 20 scientific cells and original numerical verifier in a fresh Kaggle environment. The current strengthened verifier will audit the downloaded outputs separately.

Acceptance: original source identity and configuration match; all cells complete within unchanged guards; canonical units/splits and saved numerical predictions match the original run. Report any differences, including dependency/runtime differences, without choosing the better run. Runtime and receipt timestamps need not match. Compare predictions and scientific summaries; never claim byte-identical checkpoints merely because predictions match.

This is a same-operator computational reproduction, not independent scientific replication, protected confirmation or prospective validation. Preserve failed evidence. Any discrepancy must be investigated and reported rather than retuned.
