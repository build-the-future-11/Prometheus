# Project identity

## Resolved status

**Scientific scope: `SCOPE_UNRESOLVED`. Approved mechanism: none.**

The 2026-10-19 identity audit found no qualifying local NEUROGENESIS charter, dated owner decision, paper draft, experiment manifest, or source implementation. The directory was empty before this work. Its nearest Git repository is the separate Atlas orchestration project at `/Volumes/PRO-BLADE/Atlas`, whose tracked files were already deleted in the parent working tree. Those deletions and all sibling-project work were left untouched.

The machine-readable source is `decisions/identity_audit_20261019.json`. There is no `qualifying_local_owner_source`; the field is explicitly `null`. The current owner instruction, submitted on 2026-10-19, is authoritative only for what to do when scope cannot be established: build shared benchmark/baseline infrastructure, isolate the candidate interfaces, and do not pretend a scope decision exists.

## Historical evidence preserved

Three incompatible leads are retained without merging them:

1. An undated learned-loss or learned-update-rule description reported in the current instruction.
2. An undated memory-based adaptation description, also reported in the current instruction, in which bounded memory substitutes for some parameter updates.
3. A same-day planning preview observed at 2026-10-19T06:59:48Z that labelled NEUROGENESIS “Learning through generated memory.” It was not a local owner decision artifact and did not define information timing, resource accounting, controls, or a falsifier, so it does not qualify as approval.

The learned-rule and memory-weight candidates remain separate abstract interfaces in `neurogenesis/interfaces.py`. Neither inherits from, wraps, or implements the other.

## Current project boundary

The implemented project is a mechanism-neutral adaptive-learning benchmark and evidence harness. It is not a general memory-storage toolkit, and it does not contain a candidate mechanism. The only executable methods are conventional controls admitted by `configs/protocol.json`:

- `no_adaptation`;
- two fixed SGD learning rates;
- `mechanism_off_standard_adam`, a conventional update path reserved as the future mechanism-off control.

Protected evaluation remains at zero runs. A mechanism can be added only in a separately versioned protocol after a dated local owner decision selects exactly one of:

- `APPROVED_LEARNED_RULE`; or
- `APPROVED_MEMORY_WEIGHT`.

That decision must define the mechanism, permitted information at each time, comparison set, total budget, and falsifier thresholds. A new decision must preserve this unresolved record as history.

