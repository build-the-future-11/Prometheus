"""Strict, dependency-free loaders for registered real datasets."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from datetime import date, datetime
from math import isfinite
from pathlib import Path

from .experiment import SeriesSpec


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pjm_daily(
    path: Path,
    *,
    start: date,
    end: date,
    minimum_hours_per_day: int = 20,
) -> SeriesSpec:
    """Aggregate PJM hourly MW measurements to causal daily-mean targets.

    Duplicate local timestamps are retained as observations in the daily mean. A day
    with fewer than ``minimum_hours_per_day`` finite positive readings is rejected,
    rather than silently interpolated from future data.
    """
    if start >= end:
        raise ValueError("start must precede end")
    daily: dict[date, list[float]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != 2 or "Datetime" not in reader.fieldnames:
            raise ValueError(f"unexpected PJM header in {path}")
        value_column = next(name for name in reader.fieldnames if name != "Datetime")
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.strptime(row["Datetime"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid timestamp at row {row_number} in {path}") from error
            if not start <= timestamp.date() <= end:
                continue
            try:
                value = float(row[value_column])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid value at row {row_number} in {path}") from error
            if not isfinite(value) or value <= 0:
                raise ValueError(f"non-positive or non-finite value at row {row_number}")
            daily[timestamp.date()].append(value)
    if not daily:
        raise ValueError(f"no observations in requested interval for {path}")
    values: list[float] = []
    timestamps: list[str] = []
    prior: date | None = None
    for day in sorted(daily):
        readings = daily[day]
        if len(readings) < minimum_hours_per_day:
            raise ValueError(f"day {day} has only {len(readings)} readings in {path.name}")
        if prior is not None and (day - prior).days != 1:
            raise ValueError(f"non-consecutive daily coverage between {prior} and {day}")
        prior = day
        values.append(sum(readings) / len(readings))
        timestamps.append(day.isoformat())
    if timestamps[0] != start.isoformat() or timestamps[-1] != end.isoformat():
        raise ValueError(
            f"coverage does not match requested boundaries {start} through {end} in {path.name}"
        )
    return SeriesSpec(
        name=path.stem.removesuffix("_hourly"),
        values=tuple(values),
        season=7,
        source="kaggle:robikscube/hourly-energy-consumption/" + path.name,
        source_sha256=file_sha256(path),
        timestamps=tuple(timestamps),
    )

