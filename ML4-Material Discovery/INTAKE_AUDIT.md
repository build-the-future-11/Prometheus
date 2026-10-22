# Intake audit

Date: 2026-10-19 (Asia/Kolkata)

## Repository state established before implementation

- Opened path: `/Volumes/PRO-BLADE/Atlas/ML4-Material Discovery`
- Git root resolved to the parent checkout: `/Volumes/PRO-BLADE/Atlas`
- Parent HEAD: `21b9338f23df46b32af734cb33903528a1899e8a` on `main`, matching `origin/main` at intake.
- The opened `ML4-Material Discovery` folder contained no files and no nested Git repository.
- No local charter, code, dataset, manifest, notebook, paper, model, result, or approved child task existed in the opened folder.
- The parent checkout showed tracked files outside this folder as deleted and an out-of-scope `.DS_Store` as untracked. Those changes were preserved and not restored, staged, deleted, or otherwise modified by this work.
- Sibling Atlas directories, including `Worldcastr` if present elsewhere, were out of scope and untouched.

Because the project was undefined, the explicit fallback in the user request was applied: one accessible benchmark, one primary property, calibrated property prediction, and an evaluated candidate-selection policy.

## Approved task inventory

| ID | Task | Basis | Status before work |
|---|---|---|---|
| P01 | Experimental band-gap prediction from composition | User-authorized undefined-repository fallback | NOT_STARTED |
| P01-R | Fixed-pool uncertainty-aware ranking for the same property | Required by fallback | NOT_STARTED |
| P01-G | Separately labeled element-holdout generalization checks | Appropriate composition-family study | NOT_STARTED |

No additional property, structure-generation, active-learning, electronic-structure, wet-lab, or submission work package was approved or invented. Active learning is explicitly out of scope: the chosen objective is a fixed candidate pool, not a sequential acquisition process.

## Pre-model data audit finding

The pinned Matbench artifact contains 4,604 rows but only 4,601 unique canonical compositions. Three equivalent-composition pairs have target spans of 0.58, 0.64, and 2.30 eV, contrary to the upstream metadata statement that composition duplicates with spans above 0.1 eV were removed. Exact rows and formulas are retained in `data/manifests/canonical_duplicate_audit.json`; official-fold leakage is retained in `data/manifests/official_split_leakage_audit.json`.

