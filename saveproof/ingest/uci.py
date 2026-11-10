"""UCI Steel 15-minute interval-end meter data adapter."""

from pathlib import Path

import numpy as np
import pandas as pd

from saveproof.contract import validate

_DATE_COLUMN = "date"
_KWH_COLUMN = "Usage_kWh"
_PRODUCTION_COLUMN = "Lagging_Current_Reactive.Power_kVarh"
_EXPECTED_INTERVALS_PER_DAY = 96


def _numeric_column(raw: pd.DataFrame, column: str) -> pd.Series:
    try:
        values = pd.to_numeric(raw[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"column '{column}' contains a non-numeric reading") from error

    invalid = values.lt(0) | np.isinf(values)
    if invalid.any():
        row = int(invalid.to_numpy().argmax())
        raise ValueError(
            f"column '{column}': row {row} has an invalid negative or infinite reading"
        )
    return values


def load_uci_steel(path: str | Path) -> pd.DataFrame:
    """Read UCI Steel CSV and return daily kWh and lagging-kVArh contract rows.

    CSV timestamps are interval ends in ``dd/mm/YYYY HH:MM`` format. Subtracting 15 minutes
    before taking the date assigns midnight-ending intervals to the preceding calendar day.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()
    required = [_DATE_COLUMN, _KWH_COLUMN, _PRODUCTION_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"missing UCI column(s) {missing}; found columns {raw.columns.tolist()}")
    if raw.columns.duplicated().any():
        raise ValueError("UCI CSV contains duplicate column names after trimming whitespace")

    try:
        interval_ends = pd.to_datetime(
            raw[_DATE_COLUMN], format="%d/%m/%Y %H:%M", errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"column '{_DATE_COLUMN}' must use dd/mm/YYYY HH:MM timestamps") from error
    if interval_ends.duplicated().any():
        row = int(interval_ends.duplicated().to_numpy().argmax())
        raise ValueError(f"column '{_DATE_COLUMN}': row {row} has a duplicate timestamp")
    if not interval_ends.dt.minute.isin([0, 15, 30, 45]).all():
        raise ValueError(f"column '{_DATE_COLUMN}' contains a timestamp off the 15-minute grid")

    kwh = _numeric_column(raw, _KWH_COLUMN)
    production = _numeric_column(raw, _PRODUCTION_COLUMN)
    period_start = (interval_ends - pd.Timedelta(minutes=15)).dt.normalize()
    grouped = pd.DataFrame(
        {"period_start": period_start, "kwh": kwh, "production": production}
    ).groupby("period_start", sort=True)

    daily = grouped.agg(
        kwh=("kwh", lambda values: values.sum(min_count=1)),
        production=("production", lambda values: values.sum(min_count=1)),
    )
    valid_intervals = grouped["kwh"].count()
    if (valid_intervals > _EXPECTED_INTERVALS_PER_DAY).any():
        day = valid_intervals.index[valid_intervals.gt(_EXPECTED_INTERVALS_PER_DAY)][0]
        raise ValueError(f"column '{_DATE_COLUMN}': {day.date()} has more than 96 intervals")
    if daily["kwh"].isna().any():
        day = daily.index[daily["kwh"].isna()][0]
        raise ValueError(f"column '{_KWH_COLUMN}': no valid readings on {day.date()}")

    daily = daily.reset_index()
    daily["period_end"] = daily["period_start"]
    daily["production_unit"] = "proxy:kvarh_lagging"
    daily["operating_days"] = 1.0
    daily["source"] = "real_meter"
    daily["dataset"] = "uci_steel"
    daily["coverage_pct"] = valid_intervals.to_numpy() / _EXPECTED_INTERVALS_PER_DAY * 100
    daily["event"] = None
    return validate(daily)
