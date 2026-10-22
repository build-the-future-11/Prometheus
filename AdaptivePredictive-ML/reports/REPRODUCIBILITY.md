# Reproducibility record

## Bound identities

- Source manifest archive SHA-256:
  `4548ba0486279fa45706aa61412ea4efb3bd90803996f38ab708fe7a10a53ebc`
- Result archive SHA-256:
  `742326e430bc01a15a08151edfe3c198f348e2cbfe0a17e23c1dfa34dc06e4a4`
- Run manifest SHA-256:
  `a204d446cefc2270e4fd64b1b811063c3e087158cbc413704a66deb07d114af9`
- Raw predictions SHA-256:
  `31fd1332e8861597e804e3afd125c72afcfded8a1bfc3180c075d21e1dcd1e76`
- Summary SHA-256:
  `844e33655d6d0eff2df608401a8ef91dc6711d9d85eeb88eaff2731a8dd4979e`
- Final deterministic PDF SHA-256:
  `daeacab04c4f79d6e5a4fcc75f9b9d4b1139ff09a3f7c18df4b6d1ed7e11f821`

## Environment

The hosted run used Python 3.13.15, NumPy 2.1.3, Linux x86-64, CPU only, GPU
disabled, and internet disabled. The complete hosted `pip freeze --all` output is
retained as `artifacts/kaggle-development-v2-r1/environment-v2.txt`.

## Recompute and compile

```bash
bash scripts/reproduce_paper.sh
```

The script independently verifies the raw run, regenerates every manuscript table
and PGFPlots figure from retained JSON, compiles `paper/main.pdf` with a fixed
`SOURCE_DATE_EPOCH`, and prints its SHA-256 digest. Generated-asset input and output hashes are stored in
`paper/asset-manifest.json`.

## Reproduction boundary

This is first-party hosted execution plus a separate first-party metric
recomputation. It is stronger than trusting a notebook status, but it is not an
independent researcher reproduction. Re-executing the original hosted job also
depends on access to the private Kaggle source dataset and the upstream CC0 data.
