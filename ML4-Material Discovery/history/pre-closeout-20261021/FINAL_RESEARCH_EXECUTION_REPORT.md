# Final research execution report

Date: 2026-10-20 (Asia/Kolkata)  
Branch: `codex/ml4materials-research-execution`

## Submission-readiness decision

**NOT SUBMISSION-READY.** No complete guarded experiment has produced a five-fold out-of-fold prediction, uncertainty, ranking, or element-holdout endpoint. The submission release gate rejects every retained run. The project is software- and protocol-advanced, not empirically complete.

## Implemented and verified this execution

- Repaired headless plot generation on macOS by using Matplotlib's bundled fonts (`MPL_IGNORE_SYSTEM_FONTS=1`) rather than a malformed sandboxed `system_profiler` font response. This repair preserves the Agg backend and does not change model/data/protocol behavior.
- Added `ml4materials plan`, `preflight`, and `registry`; preflight checks pinned inputs, free disk, RSS/pageout admission, and persists an explicit admission/refusal receipt before allocating a run directory.
- Added manifest-verified run indexing and a fail-closed `compare` command. Comparisons require completed, manifest-valid runs with identical source and split hashes.
- Implemented a separate direct-quantile gradient-boosting successor: deterministic median/lower/upper models, 50/80/90/95% intervals, proper interval score, atomic versioned checkpoints, reload checks, configuration, protocol, and `ml4materials run-quantile`.
- Added `ml4materials inspect-checkpoint`, which reports schema-bound ensemble feature importances with an explicit non-causal interpretation boundary. It was exercised against the retained fold-0 engineering checkpoint only.
- Added a LaTeX execution-record package, bibliography, and `scripts/reproduce_submission.py`, which rejects any incomplete or artifact-invalid run before a numerical paper export.
- Repaired a completed-run status mismatch that would have caused valid main-study runs to fail the release gate, added a positive-path gate test, and made the pageout monitor portable to Linux through `/proc/vmstat` `pswpout`.
- Added `scripts/run_kaggle.sh`, which executes the unchanged frozen study and packages only after tests, preflight, manifest verification, and the release gate pass.

## Recorded experiments

| Run | Scope | Observed result | Scientific disposition |
|---|---|---|---|
| `development-v1-r4` | Frozen three-seed ExtraTrees protocol | Fold 0 checkpoint serialized and reload-validated; peak RSS 163.9 MiB; pageouts +41 | `FAILED_RETAINED`; no aggregate metrics |
| `development-v1-r3` | Frozen three-seed ExtraTrees protocol | Preflight admitted; data validation reached 136.3 MiB peak RSS; pageouts +39 | `FAILED_RETAINED`; no aggregate metrics |
| `quantile-composition-v2-r1` | Separate direct-quantile successor | Fold 0 checkpoint serialized and reload-validated; peak RSS 160.3 MiB; pageouts +149 | `FAILED_RETAINED`; no aggregate metrics |

The no-rising-pageout condition and 512 MiB RSS ceiling were retained in both attempts. Partial fold artifacts must not be averaged, used in a baseline comparison, plotted as results, or reported as evidence of calibration or ranking utility.

## Verification receipts

- Full suite: **49 passed** in 13.61 seconds using project-volume temporary storage because the system `/tmp` volume was full.
- `paper/main.tex`: compiled successfully with the Codex LaTeX compiler.
- `uv build`: built `dist/ml4_material_discovery-0.1.0.tar.gz` and `dist/ml4_material_discovery-0.1.0-py3-none-any.whl` from the updated source.
- The release gate rejected `development-v1-r4` for `FAILED_RETAINED` status and missing complete-run outputs, as intended.
- `artifacts/preflight-20261020.json` and `artifacts/preflight-quantile-v2-20261020.json` record admitted preflights (122.7 MiB and 150.2 MiB RSS, respectively; zero pageout delta at those instants). `artifacts/run_registry-20261020-final.json` indexes retained run manifests.

## Reproducibility and paper artifacts

- Frozen protocol: `PROTOCOL.md`
- Quantile successor protocol/configuration: `PROTOCOLS/quantile-composition-v2.md`, `configs/studies/quantile-composition-v2.json`
- Retained receipts: `runs/development-v1-r4/`, `runs/quantile-composition-v2-r1/`
- Paper source: `paper/main.tex`, `paper/references.bib`
- Release gate: `scripts/reproduce_submission.py`

## Pending compute requirement

The current completed implementations are scikit-learn CPU models; a GPU would not accelerate their code path. No GPU job was submitted or claimed. A GPU job is not scientifically ready until a separate structure/graph model and a provenance-qualified, licensed structure dataset are implemented and frozen as their own study. The immediate executable command for the existing successor is:

```bash
.venv/bin/python -m ml4materials preflight --config configs/studies/quantile-composition-v2.json --output artifacts/preflight-quantile-v2.json
.venv/bin/python -m ml4materials run-quantile --run-id quantile-composition-v2-r2
```

Run it only in a stable no-pageout environment. A successful execution must then pass `ml4materials verify`, the full test suite, and `scripts/reproduce_submission.py` before numerical manuscript text, figures, tables, or a submission-readiness decision can change.
