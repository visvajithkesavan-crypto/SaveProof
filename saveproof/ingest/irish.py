"""Irish site adapter: 15-minute UTC interval-start meter files to Europe/Dublin daily rows."""

from pathlib import Path

import numpy as np
import pandas as pd

from saveproof.config import MIN_COVERAGE_PCT
from saveproof.contract import validate

_GRID_FILE = "mi_a_grid_incomer_15min_ALLphase.csv"
_PV_FILE = "mi_b_pv_generation_15min_ALLphase.csv"
_QUALITY_LOG = Path("metadata") / "meter_data_quality_log.csv"
_METER_COLUMNS = ["AssetId", "window_start_utc", "DateLocal", "Energy_kWh_15m", "IsReliableWindow"]
_LOG_COLUMNS = ["asset_id", "issue_start", "issue_end"]
_TIMEZONE = "Europe/Dublin"
_INTERVAL = pd.Timedelta(minutes=15)
_DAY = pd.Timedelta(days=1)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}")
    return raw


def _raise_at_first(bad: pd.Series, path: Path, column: str, problem: str) -> None:
    if bad.any():
        row = int(bad.to_numpy().argmax())
        raise ValueError(f"{path.name}: column '{column}': row {row} {problem}")


def _parse_times(raw: pd.DataFrame, path: Path, column: str, fmt: str) -> pd.Series:
    try:
        return pd.to_datetime(raw[column], format=fmt, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path.name}: column '{column}' must use format {fmt}") from error


def _read_meter(path: Path, asset_id: str) -> pd.DataFrame:
    """Return one meter's windows: local_day (Dublin date, tz-naive), kwh, reliable (0/1)."""
    raw = _read_csv(path, _METER_COLUMNS)
    _raise_at_first(raw["AssetId"] != asset_id, path, "AssetId", f"is not {asset_id!r}")
    start_utc = _parse_times(raw, path, "window_start_utc", "%Y-%m-%d %H:%M:%S")
    _raise_at_first(start_utc.duplicated(), path, "window_start_utc", "is a duplicate")
    off_grid = start_utc != start_utc.dt.floor(_INTERVAL)
    _raise_at_first(off_grid, path, "window_start_utc", "is off the 15-minute grid")

    local_day = (
        start_utc.dt.tz_localize("UTC").dt.tz_convert(_TIMEZONE).dt.tz_localize(None).dt.normalize()
    )
    date_local = _parse_times(raw, path, "DateLocal", "%Y-%m-%d")
    mismatch = date_local != local_day
    _raise_at_first(mismatch, path, "DateLocal", "differs from window_start_utc in Europe/Dublin")

    kwh = pd.to_numeric(raw["Energy_kWh_15m"], errors="coerce")
    bad_kwh = kwh.isna() | kwh.lt(0) | np.isinf(kwh)
    _raise_at_first(bad_kwh, path, "Energy_kWh_15m", "is not a non-negative number")
    reliable = raw["IsReliableWindow"]
    _raise_at_first(~reliable.isin([0, 1]), path, "IsReliableWindow", "is not 0 or 1")
    return pd.DataFrame({"local_day": local_day, "kwh": kwh, "reliable": reliable})


def _daily_totals(windows: pd.DataFrame) -> pd.DataFrame:
    """Per local day: windows present, kWh summed, reliable windows counted."""
    return windows.groupby("local_day", sort=True).agg(
        windows=("kwh", "size"), kwh=("kwh", "sum"), reliable=("reliable", "sum")
    )


def _day_bounds_utc(local_days: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Start and end (exclusive) of each Dublin local day, as tz-naive UTC."""
    starts = local_days.tz_localize(_TIMEZONE).tz_convert("UTC").tz_localize(None)
    ends = (local_days + _DAY).tz_localize(_TIMEZONE).tz_convert("UTC").tz_localize(None)
    return starts, ends


def _fault_days(path: Path, local_days: pd.DatetimeIndex) -> np.ndarray:
    """True for local days overlapping any logged issue interval of mi_a or mi_b."""
    log = _read_csv(path, _LOG_COLUMNS)
    log = log[log["asset_id"].isin(["mi_a", "mi_b"])].reset_index(drop=True)
    # TODO(verify): the log does not state its timezone; read as UTC like the meter files
    # (dataset README: "Timestamps are in UTC").
    issue_starts = _parse_times(log, path, "issue_start", "%d/%m/%Y %H:%M")
    issue_ends = _parse_times(log, path, "issue_end", "%d/%m/%Y %H:%M")
    _raise_at_first(issue_ends < issue_starts, path, "issue_end", "is before issue_start")

    day_starts, day_ends = _day_bounds_utc(local_days)
    faulty = np.zeros(len(local_days), dtype=bool)
    for issue_start, issue_end in zip(issue_starts, issue_ends, strict=True):
        faulty |= (day_starts < issue_end) & (day_ends > issue_start)
    return faulty


def load_irish(data_dir: str | Path) -> pd.DataFrame:
    """Read the Irish grid (mi_a) and PV (mi_b) meters; return Dublin-local daily contract rows.

    data_dir is data/raw/irish. kwh is grid import summed per local day; production is same-day
    PV kWh (NaN when reliable PV windows cover < MIN_COVERAGE_PCT of the day); coverage_pct counts
    reliable grid windows against the day's length (92/96/100 windows). Meters are hierarchical
    and are never summed together. Rows start at the first complete grid day.
    """
    data_dir = Path(data_dir)
    grid = _daily_totals(_read_meter(data_dir / _GRID_FILE, "mi_a"))
    day_starts, day_ends = _day_bounds_utc(grid.index)
    expected = pd.Series((day_ends - day_starts) / _INTERVAL, index=grid.index)
    too_many = grid["windows"] > expected
    if too_many.any():
        day = grid.index[too_many.to_numpy()][0]
        raise ValueError(f"{_GRID_FILE}: {day.date()} has more windows than the local day holds")
    complete = grid["windows"] == expected
    if not complete.any():
        raise ValueError(f"{_GRID_FILE}: no complete local day")
    first_complete = complete.idxmax()
    grid, expected = grid.loc[first_complete:], expected.loc[first_complete:]

    pv = _daily_totals(_read_meter(data_dir / _PV_FILE, "mi_b")).reindex(grid.index)
    pv_coverage_pct = 100 * pv["reliable"].fillna(0) / expected
    daily = pd.DataFrame(
        {
            "period_start": grid.index,
            "period_end": grid.index,
            "kwh": grid["kwh"].to_numpy(),
            "production": pv["kwh"].where(pv_coverage_pct >= MIN_COVERAGE_PCT).to_numpy(),
            "production_unit": "proxy:pv_kwh",
            "operating_days": 1.0,
            "source": "real_meter",
            "dataset": "irish_mi_a",
            "coverage_pct": (100 * grid["reliable"] / expected).to_numpy(),
            "event": np.where(
                _fault_days(data_dir / _QUALITY_LOG, grid.index), "meter_fault", None
            ),
        }
    )
    return validate(daily)
