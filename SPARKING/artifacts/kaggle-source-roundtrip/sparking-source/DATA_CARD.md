# Dataset card: Matbench experimental band gap

## Identity and provenance

- Dataset: `matbench_expt_gap`, Matbench v0.1.
- Canonical URL: `https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz`.
- Required SHA-256: `783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2`.
- Rows: 4,604; canonical compositions under local scale/order invariance: 4,601.
- Input: chemical composition string.
- Target: experimentally measured inorganic-material band gap in eV.
- Observed local range: 0.0--11.7 eV.
- Upstream source/curation: Matbench and the Matminer experimental-gap dataset.
- License: the upstream Figshare source-data landing page declares MIT; the
  Materials Project mirror does not embed a machine-readable license in the
  compressed artifact.

`sparking prepare` downloads the artifact, refuses any digest mismatch, parses
every formula, validates every target, and regenerates the official five folds.
The canonical regenerated fold manifest must hash to
`69bc38097c78dd752cdc5506d21b08f32bc3def921b2aac6fc3c8681a1adb566`.

## Units and independent experimental units

The row-level target is in eV. The primary confirmatory units are five declared
element-exclusion scenarios (`Li`, `Zn`, `Ga`, `Se`, `Te`), not the thousands
of overlapping rows. Element scenarios are still not statistically independent
because a composition may contain more than one held-out element; all five
effects are therefore reported individually.

## Splits

- Development only: leave-one-element-out for `Cu`, `Ag`, and `B`.
- Protected confirmation: leave-one-element-out for `Li`, `Zn`, `Ga`, `Se`,
  and `Te`.
- Compatibility endpoint: official Matbench five-fold splits regenerated with
  `KFold(n_splits=5, shuffle=True, random_state=18012019)`.
- Interval calibration: within each official training fold only, a deterministic
  80/20 proper-training/calibration split. Outer-test rows never calibrate.

## Known gaps and prohibited uses

Composition does not identify crystal structure, phase, polymorph, processing
history, defects, measurement protocol, thermodynamic stability, or synthesis
feasibility. Three canonical duplicate groups are present upstream, and the
predecessor audit found canonical-composition train/test overlap in official
folds 0--3. Official results are comparability evidence, not extrapolation
evidence. The dataset must not be used here to claim a newly discovered,
stable, synthesizable, safe, or useful material.

