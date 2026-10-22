# Final research execution report

Date: 2026-10-20  
Branch: `codex/neurogenesis-research-execution`

## Decision

**NOT SUBMISSION READY for a mechanism-efficacy paper.** The v1 baseline benchmark is now accompanied by a verified analysis and publication package, but the core NEUROGENESIS mechanism remains `SCOPE_UNRESOLVED`. No learned-rule or memory-weight candidate was selected, implemented, or evaluated. Protected evaluation remains at zero runs.

## Executed and verified

1. Preserved the v1 execution package and every retained experiment artifact. No file under `neurogenesis/` was changed, so the recorded producer source identity remains valid.
2. Re-ran the v1 independent verifier after publication-tool work. It returned `VALID`: 20 verified attempts, 9,600 verified trace rows, zero protected-evaluation runs, 84 retained artifacts, and artifact-map SHA-256 `f353206d36fd03ec596a2b4006ef181715c524c43ec13e53d07e2ec28dc089d6`.
3. Added `scripts/analyze_bundle.py`, a read-only standard-library analysis pipeline. It recomputes all summaries from retained `result.json` and `trace.jsonl` files; records input hashes; exports JSON, CSV, LaTeX, offline HTML, rolling-MSE SVG, and a stability--plasticity SVG.
4. Added `scripts/verify_analysis.py`. It regenerates all seven derived artifacts and compares them byte-for-byte. The accepted receipt is `analysis_receipts/development-baselines-v1-20261020-r2-verification.json` (`VALID`; derived-artifact-map SHA-256 `30e8a963ac721bbf4013c71c0c25f2bc549ebe75df85591f850f66b45946bc87`).
5. Added `scripts/assemble_submission.py` and produced `submission/neurogenesis-v1-development-baselines-20261020-r2/`, containing a LaTeX paper, bibliography stub, vector figures, tables, raw traces/checkpoints, source, configs, reproduction files, analysis, manifest, GPU-status statement, and submission-readiness decision.
6. Compiled the generated LaTeX source successfully with Tectonic to `/tmp/neurogenesis-paper-build-r2/paper.pdf`.
7. Ran the full unit suite: 18 tests passed, including analysis regression tests that recompute the known baseline result from raw traces and reject output overwrites.

## Verified results

| Policy | Overall MSE, mean ± sample SD | Switch MSE | Return MSE |
|---|---:|---:|---:|
| Frozen | 1.0110 ± 0.0671 | 0.9611 | 1.2241 |
| SGD 0.01 | 0.9484 ± 0.0627 | 1.9652 | 2.2782 |
| SGD 0.05 | 3.3480 ± 0.2161 | 6.7383 | 12.2006 |
| Adam 0.01 | 0.9978 ± 0.0943 | 1.9225 | 1.9003 |

The paired development difference for SGD 0.01 against frozen prediction was negative for all five fixed identities (mean −0.0626). This is descriptive development evidence only: it establishes neither a population inference nor a mechanism claim. Its worse switch and return metrics demonstrate the benchmark's stability--plasticity tradeoff.

## Skeptical audit findings

- The central mechanism is absent by design because no dated owner decision selects a learned-rule or memory-weight hypothesis. Implementing either would make an unauthorized scientific choice.
- The existing results are synthetic, linear, small (five development identities), and development-only. They do not establish generalization, robustness, real-stream performance, independent replication, or publication readiness for a mechanism.
- The derived verifier initially found an invalid path comparison when it regenerated an absolute bundle path against a saved relative path. The defect was repaired; the original invalid receipt was retained, and the second receipt is valid.
- There is no valid GPU workload in the admitted v1 scope. The recorded baseline matrix is CPU-only and completes in milliseconds per run. Candidate GPU jobs remain blocked by the missing mechanism decision and protocol-v2 admission.

## Concrete next gate

Before candidate architecture, GPU work, ablations, or protected evaluation, add a dated decision selecting exactly one candidate, then freeze a separately versioned protocol v2 with candidate information flow, resource ledger, development search rule, matched controls, and falsifier thresholds. The retained v1 result and this report must remain historical evidence.
