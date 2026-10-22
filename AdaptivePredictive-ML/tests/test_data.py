from __future__ import annotations

from datetime import date, timedelta

import pytest

from adaptive_predictive.data import load_pjm_daily


def write_fixture(path, *, days: int = 20, hours: int = 24) -> None:
    lines = ["Datetime,TEST_MW"]
    start = date(2020, 1, 1)
    for offset in range(days):
        day = start + timedelta(days=offset)
        for hour in range(hours):
            lines.append(f"{day.isoformat()} {hour:02d}:00:00,{100 + offset + hour / 10}")
    path.write_text("\n".join(lines) + "\n")


def test_load_pjm_daily_is_sorted_and_aggregated(tmp_path) -> None:
    path = tmp_path / "TEST_hourly.csv"
    write_fixture(path)
    spec = load_pjm_daily(path, start=date(2020, 1, 1), end=date(2020, 1, 20))
    assert len(spec.values) == 20
    assert spec.timestamps is not None
    assert spec.timestamps[0] == "2020-01-01"
    assert spec.values[0] == pytest.approx(sum(100 + hour / 10 for hour in range(24)) / 24)


def test_load_pjm_daily_rejects_incomplete_day(tmp_path) -> None:
    path = tmp_path / "TEST_hourly.csv"
    write_fixture(path, hours=19)
    with pytest.raises(ValueError, match="only 19 readings"):
        load_pjm_daily(path, start=date(2020, 1, 1), end=date(2020, 1, 20))


def test_load_pjm_daily_does_not_parse_protected_out_of_window_values(tmp_path) -> None:
    path = tmp_path / "TEST_hourly.csv"
    write_fixture(path)
    with path.open("a") as handle:
        handle.write("2021-01-01 00:00:00,PROTECTED_INVALID_VALUE\n")
    spec = load_pjm_daily(path, start=date(2020, 1, 1), end=date(2020, 1, 20))
    assert len(spec.values) == 20


def test_load_pjm_daily_rejects_missing_boundary_days(tmp_path) -> None:
    path = tmp_path / "TEST_hourly.csv"
    write_fixture(path, days=19)
    with pytest.raises(ValueError, match="coverage does not match requested boundaries"):
        load_pjm_daily(path, start=date(2020, 1, 1), end=date(2020, 1, 20))

