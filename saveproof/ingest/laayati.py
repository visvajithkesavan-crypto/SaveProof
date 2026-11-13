"""Laayati et al. (2022) open-pit mine, hourly per-station data adapter.

Source and preparation rules: docs/validation_addendum_v5b.md §3-§4. Timestamps are plant-local
(time zone not stated; Morocco has no DST change in the period) and mark the START of each hour.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from saveproof.contract import validate

STATIONS = ("destoning", "screening", "loading")
_TIME_COLUMN = "Time"
_KWH_COLUMN = "KWh"
_PRODUCTION_COLUMN = "THE"  # production KPI from the process SCADA; unit undocumented (§3)
_HOURS_PER_DAY = 24


def _numeric_column(raw: pd.DataFrame, column: str) -> pd.Series:
    try:
        values = pd.to_numeric(raw[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"column '{column}' contains a non-numeric reading") from error
    invalid = values.lt(0) | np.isinf(values)
    if invalid.any():
        row = int(invalid.to_numpy().argmax())
        raise ValueError(f"column '{column}': row {row} is negative or infinite")
    return values


def _hour_starts(raw: pd.DataFrame) -> pd.Series:
    """Parsed d/m/yy H:MM timestamps; raises unless each day has 24 distinct on-the-hour rows."""
    try:
        stamps = pd.to_datetime(raw[_TIME_COLUMN], format="%d/%m/%y %H:%M", errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"column '{_TIME_COLUMN}' must use d/m/yy H:MM timestamps") from error
    off_hour = stamps != stamps.dt.floor("h")
    if off_hour.any():
        row = int(off_hour.to_numpy().argmax())
        raise ValueError(f"column '{_TIME_COLUMN}': row {row} is not on the hour")
    hours_per_day = stamps.dt.normalize().value_counts().sort_index()
    wrong = hours_per_day[hours_per_day != _HOURS_PER_DAY]
    if not wrong.empty:
        raise ValueError(
            f"column '{_TIME_COLUMN}': {wrong.index[0].date()} has {wrong.iloc[0]} hours, "
            f"expected {_HOURS_PER_DAY}"
        )
    if stamps.duplicated().any():
        row = int(stamps.duplicated().to_numpy().argmax())
        raise ValueError(f"column '{_TIME_COLUMN}': row {row} is a duplicate timestamp")
    return stamps


def load_laayati(path: str | Path, station: str) -> pd.DataFrame:
    """Read one station's hourly CSV; return daily contract rows (kWh, THE KPI, plant-local dates).

    kwh = sum of present KWh hours, coverage_pct = 100 x present KWh hours / 24. production = sum
    of THE, NaN if any hour of THE is missing (never a partial sum).
    """
    if station not in STATIONS:
        raise ValueError(f"station must be one of {STATIONS}, got {station!r}")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()  # the real header is "Time,THE,HM ,KWh"
    required = [_TIME_COLUMN, _PRODUCTION_COLUMN, _KWH_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"missing Laayati column(s) {missing}; found {raw.columns.tolist()}")

    grouped = pd.DataFrame(
        {
            "period_start": _hour_starts(raw).dt.normalize(),
            "kwh": _numeric_column(raw, _KWH_COLUMN),
            "production": _numeric_column(raw, _PRODUCTION_COLUMN),
        }
    ).groupby("period_start", sort=True)
    daily = grouped.agg(
        kwh=("kwh", lambda values: values.sum(min_count=1)),
        production=("production", lambda values: values.sum(skipna=False)),
    )
    if daily["kwh"].isna().any():
        day = daily.index[daily["kwh"].isna()][0]
        raise ValueError(f"column '{_KWH_COLUMN}': no readings on {day.date()}")

    kwh_hours = grouped["kwh"].count().to_numpy()
    daily = daily.reset_index()
    daily["period_end"] = daily["period_start"]
    daily["production_unit"] = "process_kpi:THE"
    daily["operating_days"] = 1.0
    daily["source"] = "real_meter"
    daily["dataset"] = f"laayati_{station}"
    daily["coverage_pct"] = 100 * kwh_hours / _HOURS_PER_DAY
    daily["event"] = None
    return validate(daily)
