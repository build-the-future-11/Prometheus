# Model card: robust aggregation v3

**Intended use:** research prediction of experimental inorganic band gap in eV
from formula alone, within the tested historical composition distribution.
**Artifact:** `runs/kaggle-robust-aggregation-v3-r1/models/` (15 development outer
folds and five element-holdout checkpoints). There is no silently retrained
all-data model or independent production release.

The fixed 128-feature representation feeds three 96-tree ExtraTrees ensembles.
The default point estimate is the median of individual tree leaf means. A saved
calibration score distribution gives normalized residual intervals, with a
predeclared .05 eV dispersion floor. These are not conditional-density estimates.

The grouped three-repeat mean MAE is 0.425622 eV versus 0.473870 eV for mean
aggregation of exactly the same trees. The primary improvement is supported
within development scope. RMSE and positive-target MAE worsen. Element shifts
remain poorly calibrated, especially selenium. See the full subgroup, calibration
and holdout tables; do not select only the favorable endpoint.

Training uses 4,601 canonical groups derived from 4,604 licensed source rows,
with mean targets for conflicting duplicates and preserved source labels.
Data, source, environment and model hashes accompany the run. The checkpoint
schema is `robust-aggregation-v3:1`; trusted joblib loading is required.

Limitations: formula ambiguity, missing polymorph/measurement conditions,
large zero-gap mass, no protected external test set, no independent reproduction,
no prospective validation, no neural/SOTA comparison. A narrow interval is not
proof of OOD accuracy. Predictions establish neither stability nor synthesis,
safety, application utility or new-material discovery.
