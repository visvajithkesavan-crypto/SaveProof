"""The data contract: the one table every layer speaks (columns: see project rules)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SOURCES = {"real_meter", "utility_bill", "simulated"}
EVENTS = {"retrofit", "shutdown", "meter_fault", "holiday"}
REQUIRED = [
    "period_start",
    "period_end",
    "kwh",
    "production",
    "production_unit",
    "operating_days",
    "source",
    "dataset",
    "coverage_pct",
    "event",
]
PERIOD_COLUMNS = ["period_start", "period_end"]
NUMERIC_COLUMNS = ["kwh", "production", "operating_days", "coverage_pct"]
# Columns that must hold exactly one value per frame (one plant at a time).
SINGLE_VALUE_COLUMNS = ["dataset", "production_unit"]
# production and event are the only nullable columns.
NOT_NULL = [c for c in REQUIRED if c not in ("production", "event")]
PLANT_DATE_FIELDS = [
    "baseline_start",
    "baseline_end",
    "reporting_start",
    "reporting_end",
    "retrofit_date",
]


@dataclass(frozen=True)
class PlantMeta:
    plant_id: str
    sector: str
    timezone: str
    baseline_start: pd.Timestamp
    baseline_end: pd.Timestamp
    reporting_start: pd.Timestamp
    reporting_end: pd.Timestamp
    retrofit_date: pd.Timestamp | None

    def __post_init__(self) -> None:
        # Timestamps (not datetime.date) so they compare directly with datetime64 columns.
        for name in PLANT_DATE_FIELDS:
            value = getattr(self, name)
            if value is None and name == "retrofit_date":
                continue
            if not isinstance(value, pd.Timestamp) or value.tzinfo is not None:
                raise ValueError(f"{name} must be a tz-naive pd.Timestamp, got {value!r}")
            if value != value.normalize():
                raise ValueError(f"{name} must be at midnight, got {value!r}")
        if not self.baseline_start <= self.baseline_end:
            raise ValueError("baseline_end must be on or after baseline_start")
        if not self.baseline_end < self.reporting_start:
            raise ValueError("reporting_start must be after baseline_end")
        if not self.reporting_start <= self.reporting_end:
            raise ValueError("reporting_end must be on or after reporting_start")
        if self.retrofit_date is not None and not (
            self.baseline_end < self.retrofit_date <= self.reporting_start
        ):
            raise ValueError("retrofit_date must be after baseline_end, at most reporting_start")


def _raise_at_first(df: pd.DataFrame, bad: pd.Series, column: str, problem: str) -> None:
    """Raise ValueError naming column and first row (df index label) where bad is True.

    problem may contain {value}, replaced by the offending cell.
    """
    if bad.any():
        position = bad.to_numpy().argmax()
        message = problem.format(value=df[column].iloc[position])
        raise ValueError(f"column '{column}': row {df.index[position]} {message}")


def _check_types_and_nulls(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")
    for column in PERIOD_COLUMNS:
        if not pd.api.types.is_datetime64_dtype(df[column]):
            raise ValueError(
                f"column '{column}': dtype is {df[column].dtype}, expected datetime64 (no timezone)"
            )
    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"column '{column}': dtype is {df[column].dtype}, expected numeric")
    for column in NOT_NULL:
        _raise_at_first(df, df[column].isna(), column, "is null")
    for column in PERIOD_COLUMNS:
        _raise_at_first(df, df[column] != df[column].dt.normalize(), column, "is not at midnight")


def _check_values(df: pd.DataFrame) -> None:
    for column in NUMERIC_COLUMNS:  # NaN in nullable production is allowed; inf never is
        _raise_at_first(df, np.isinf(df[column]), column, "is infinite ({value})")
    unknown_source = ~df["source"].isin(SOURCES)
    _raise_at_first(df, unknown_source, "source", f"is {{value!r}}, not in {sorted(SOURCES)}")
    unknown_event = df["event"].notna() & ~df["event"].isin(EVENTS)
    _raise_at_first(df, unknown_event, "event", f"is {{value!r}}, not in {sorted(EVENTS)}")
    _raise_at_first(df, df["kwh"] < 0, "kwh", "is negative ({value})")
    _raise_at_first(df, df["production"] < 0, "production", "is negative ({value})")
    bad_coverage = ~df["coverage_pct"].between(0, 100)
    _raise_at_first(df, bad_coverage, "coverage_pct", "is outside 0-100 ({value})")
    for column in SINGLE_VALUE_COLUMNS:  # df is non-empty here, so iloc[0] is safe
        other_value = df[column] != df[column].iloc[0]
        _raise_at_first(df, other_value, column, f"is {{value!r}}; validate one {column} at a time")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check df (one plant) against the contract; return a copy sorted by period.

    Raises ValueError naming the column and first bad row; never fixes data.
    """
    if len(df) == 0:
        raise ValueError("no rows")
    _check_types_and_nulls(df)
    _check_values(df)

    # period_end is inclusive, so a one-day row has start == end; only end < start is an error.
    _raise_at_first(
        df, df["period_end"] < df["period_start"], "period_end", "is before period_start"
    )
    period_days = (df["period_end"] - df["period_start"]).dt.days + 1
    bad_days = ~df["operating_days"].between(0, period_days)
    _raise_at_first(df, bad_days, "operating_days", "is outside 0..days in period ({value})")

    ordered = df.sort_values(PERIOD_COLUMNS, kind="stable")
    overlaps_previous = ordered["period_start"] <= ordered["period_end"].shift()
    _raise_at_first(ordered, overlaps_previous, "period_start", "overlaps the previous period")
    return ordered.reset_index(drop=True)
