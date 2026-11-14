# AdaptivePredictive: configuration admission repair

Date: 14 November 2026. Reference commit:
`22461b231d4c44fc41562bf24b128a67478524b3`.

## What changed

The model admitted invalid configurations before any experimental run started.
Examples include NaN ridge or learning-rate values, zero or fractional interval
windows, empty or noninteger lags, negative clipping limits, and negative scale
floors. Some fail later; others silently contaminate calculations or change the
intended semantics. Configuration validation now requires finite real-valued
hyperparameters, positive scale/clipping limits, nonnegative drift gain, positive
integer interval windows and seasons, and nonempty increasing positive integer
lags. Boolean values are not accepted as numbers.

This change leaves forecasting formulas, adaptation policies, and scientific
thresholds unchanged. It does not turn the negative DDFS comparison into a
positive result.

## Executed evidence

- The complete available AdaptivePredictive test suite passed: **101 tests**.
- **16 invalid cases** previously admitted are now rejected at construction.
- An independent comparison loads the exact baseline model from the pinned
  Git commit and checks the repaired model on **63 combinations** of three
  configurations, three seasons, and seven aggregation modes.
- **8,064 synthetic forecasts** and their expert coefficients, covariance
  matrices, and aggregation weights match exactly, with **zero mismatches**.
- Streams include a fixed level shift, sinusoidal structure, noise, and isolated
  outliers. They are a deterministic regression check, not a forecasting
  efficacy benchmark or a universal numerical proof.

The historical development gate and protected-data boundary remain unchanged.
The test runner emitted one unrelated pytest-plugin deprecation warning.

## Reproduction

From the repository root, with NumPy and pytest installed:

```bash
PYTHONPATH=AdaptivePredictive-ML/src python -m pytest AdaptivePredictive-ML/tests -q
PYTHONPATH=AdaptivePredictive-ML/src python AdaptivePredictive-ML/scripts/verify_configuration_admission.py --output AdaptivePredictive-ML/artifacts/configuration-admission-verification-20261114.json
```

The verifier requires the pinned baseline Git object, records baseline and
patched source digests, and emits the full 63-case comparison record.
