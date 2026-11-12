# Derived Learning package repair: 12 November 2026

The `Derived-Learning-ML` package failed test collection on Python 3.12 because
its dynamic test loader executed a dataclass-containing module before putting
that module into `sys.modules`. The same defect was retained inside the
previous uploadable source bundle. This repair integrates the loader correction
into the reproducibility branch and creates a separate source package revision.

## Completed verification

- The original checkout reproduced the import failure: 23 unit tests passed
  and collection of the seven S3 checks failed with a dataclass import error.
- The repaired checkout passes all 30 existing tests.
- A clean extraction of the new source bundle discovers 30 tests: 28 pass and
  the two checks requiring outer release/Kaggle metadata skip as designed.
- The existing project checker passes all eight stages: build dependencies,
  direct wheel and source archive, wheel from source archive, installation,
  dependency consistency, installed import, unit tests, and environment receipt.
- Direct and source-archive-derived wheel member hashes are identical.
- A repeated build request for an existing release is refused and the archive
  bytes remain unchanged.

The original source bundle remains at `Derived-Learning-ML/releases/s3-r0/`.
The new package and matching sidecar are at `Derived-Learning-ML/releases/s3-r1/`.
The scientific protocol remains `DLML-S3-R0`; `s3-r1` names an engineering
package revision, not another scientific study or completed hosted run.

## Evidence and limits

`derived-repair-receipt.json` records source hashes and distinguishes the base
commit from the repaired worktree that was tested. `derived-package/receipt.json`
and its logs contain the package-check evidence. `clean-extraction-tests.log`
contains the isolated archive result. No new test has been invented merely to
inflate these totals; these are the existing project suites.

The frozen model implementation, verifier, real-data configuration, protocol,
hosted launcher, and previous bundle are byte-identical to the branch base.
No protected data were opened. No GPU training or Kaggle upload was performed.
Remote Kaggle state was not rechecked. The private source update and the
registered real-data study are still separate unfinished work.

The repository inventory remains incomplete: four root gitlinks have no
`.gitmodules` mapping, and the tracked META-RENN project does not establish a
complete Python implementation. The existing source-completeness gate remains
unchanged. This repair makes no whole-portfolio completion claim.
