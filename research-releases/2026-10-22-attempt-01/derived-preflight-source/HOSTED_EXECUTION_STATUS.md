# Hosted execution status — DLML-S3-R0

Updated: 2026-10-22

## Current state: REMOTE_SOURCE_UPDATE_REQUIRED

The frozen real-data successor has not started and has no protected-test
result. The private Kaggle dataset version 2 and notebook remain historical
handoff evidence, but version 2 must not be submitted.

A clean-room extraction audit found that the v2 archive contained macOS
AppleDouble metadata and omitted `configs/smoke.json` plus the non-S3 protocol
files required by the launcher's full preflight suite. The hosted launcher
would therefore fail at host step 0 before GPU training. No protected target
was opened, no unrelated run was stopped, and the scientific protocol was not
retuned.

The local replacement source artifact now:

- excludes and rejects `._*`, `.DS_Store`, unsafe paths, links, and devices;
- contains every file required by the full 30-test preflight suite;
- passes that suite from an exact clean extraction (two outer-release-only
  checks skip inside the nested payload as designed);
- builds `derived_learning_ml-0.1.0-py3-none-any.whl` offline from that
  extraction; and
- includes an independent verifier for the returned archive, sidecar,
  manifest, failure denominator, raw metrics, and registered classification.

## Remote resources and pinned settings

- Private source dataset: <https://www.kaggle.com/datasets/ryangomez010/derived-learning-s3-frozen-source>
- Historical uploaded source version: 2 (do not submit)
- Required next uploaded source version: 3 or later, containing the corrected
  local bundle and its matching sidecar
- Private notebook: <https://www.kaggle.com/code/ryangomez010/derived-learning-s3-real-data-gpu-r1/edit>
- Required accelerator: GPU T4 x2
- Required internet setting: off
- Required attached data: corrected private frozen source and public
  `camnugent/california-housing-prices`
- Corrected local source-bundle SHA-256: `fb24189f110a430f9868acc430aa487764cb6ab30a950dd00126b3bd139f5f86`
- Hosted launcher SHA-256: `7ede71b2a1ca69813fd890bb0b35514ce8987d3bc73e6ffb466c2d27bb44630f`
- Frozen study implementation SHA-256: `0407b6d9f34b3c2997a27e2f69fe289a8cfa42d97b95ba4d76cebcba1cdc2970`

The corrected local source-bundle SHA-256 is also recorded in the checked-in
`releases/s3-r0/dlml-s3-source.sha256` sidecar. The private dataset must be
updated and the notebook re-saved against that exact sidecar before `Save & Run
All`. Kaggle authentication is currently unavailable in the connected Chrome
profile, so the remote update has not been performed or claimed.

## Submission and verification gate

After the corrected private source version is attached, re-verify privacy,
GPU T4 x2, internet off, both data attachments, the hash-checking launcher,
and the sidecar hash. Then create `Save & Run All` version `Frozen S3 R0 full
GPU run r1`. Preserve every failed candidate/test cell and traceback.

On completion, download both the archive and `.sha256` sidecar and run
`scripts/verify_s3_hosted_archive.py`. Only a `VERIFIED` receipt may admit the
raw `SUPPORTED`, `MIXED`, `NEGATIVE`, or `INCONCLUSIVE` classification. An
`INVALID` receipt admits no scientific result.

## Claim boundary

`REMOTE_SOURCE_UPDATE_REQUIRED` is engineering and provenance evidence only.
It is not a scientific result, successful hosted execution, publication-ready
result, or independent reproduction. The terminal scientific classification
remains unavailable until the protected run completes and its downloaded
evidence verifies.
