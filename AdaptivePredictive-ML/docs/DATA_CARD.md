# Development dataset card

## Identity and permitted use

- Registry identity: `kaggle:robikscube/hourly-energy-consumption`
- Dataset title: *Hourly Energy Consumption*
- Stated content: more than ten years of hourly PJM-region electricity
  consumption in megawatts.
- Protocol-declared license: CC0 1.0.
- Intended use: retrospective, one-step-ahead causal evaluation of lightweight
  univariate online forecasters.
- Prohibited interpretation: the ten registered files are correlated regional
  series, not ten independent domains or a representative sample of all grids.

The exact Kaggle dataset version, download receipt, and per-file hashes have not
yet been captured in this workspace. The protocol cannot be frozen until the
hosted mount supplies those receipts. Public dataset-page metadata was checked on
2026-10-21, but that page view is not a substitute for a versioned execution
manifest.

## Registered files

`AEP_hourly.csv`, `COMED_hourly.csv`, `DAYTON_hourly.csv`, `DEOK_hourly.csv`,
`DOM_hourly.csv`, `DUQ_hourly.csv`, `EKPC_hourly.csv`, `FE_hourly.csv`,
`PJME_hourly.csv`, and `PJMW_hourly.csv`.

Files were selected before real-data outcomes were inspected. A missing file is a
failed registered cell, not permission to substitute another series after seeing
results.

## Temporal roles

- Model input/warm-up: 2015-01-01 through 2015-12-31.
- Development evaluation: 2016-01-01 through 2016-12-31.
- Protected confirmation begins: 2017-01-01.

The development loader skips out-of-window rows before parsing their values. It
still hashes the full source file for provenance; that digest does not reveal or
evaluate protected outcomes. Confirmation requires a separate frozen runner and
receipt and must never be launched from the development script.

## Transformation

Rows must have exactly one `Datetime` column and one measurement column. Timestamps
are parsed as local clock readings in the source representation. For each calendar
day, all finite positive observations are averaged. Days with fewer than 20 values
are rejected; missing days are rejected; no interpolation occurs. The minimum-count
rule tolerates daylight-saving transitions but may also admit up to four absent
readings, so missingness counts must be reported from the hosted manifest before
freeze.

## Known validity risks

1. Daily averaging discards intraday structure and turns this into a daily-load
   study, not a short-term hourly-load benchmark.
2. The files cover related U.S. electricity regions and cannot establish
   cross-domain generalization.
3. Calendar, weather, holiday, and economic covariates are intentionally absent;
   claims concern univariate adaptation only.
4. Source timestamps may contain daylight-saving duplicates or gaps. Duplicate
   clock readings are retained in the daily mean; the per-day count must be audited.
5. MASE uses a training-only seasonal-naive denominator with season seven. Zero or
   non-finite scale invalidates the affected series.
6. Dataset licensing and version identity must be rechecked at execution time and
   bound into the freeze receipt.
