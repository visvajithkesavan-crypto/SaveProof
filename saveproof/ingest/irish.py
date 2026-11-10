"""Irish 15-minute UTC meter data adapter using Europe/Dublin local days."""

from pathlib import Path

import numpy as np
import pandas as pd

from saveproof.contract import validate

_TIME_COLUMN = "window_start_utc"
_CONSUMPTION_COLUMN = "mi_a"
_PRODUCTION_COLUMN = "mi_b"
_QUALITY_LOG = "meter_data_quality_log.csv"
_TIMEZONE = "Europe/Dublin"
_INTERVAL = pd.Timedelta(minutes=15)


def _resolve_paths(path: str | Path) -> tuple[Path, Path]:
    source = Path(path)
    if source.is_dir():
        if (source / "extracted").is_dir():
            dataset_root = source
            data_dir = source / "extracted"
        else:
            data_dir = source
            dataset_root = source.parent if source.name == "extracted" else source
        matches = sorted(data_dir.glob("*_15min_ALLphase.csv"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one *_15min_ALLphase.csv in {data_dir}, found {len(matches)}"
            )
        source = matches[0]
    else:
        dataset_root = source.parent.parent if source.parent.name == "extracted" else source.parent

    candidates = [
        dataset_root / "metadata" / _QUALITY_LOG,
        dataset_root / _QUALITY_LOG,
        source.parent / _QUALITY_LOG,
    ]
    quality_log = next((candidate for candidate in candidates if candidate.is_file()), None)
    if quality_log is None:
        raise FileNotFoundError(
            f"required Irish meter quality log '{_QUALITY_LOG}' not found near {source}"
        )
    return source, quality_log


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


def _expected_intervals(local_day: pd.Timestamp) -> int:
    next_day = local_day + pd.DateOffset(days=1)
    duration = next_day.tz_convert("UTC") - local_day.tz_convert("UTC")
    return int(duration / _INTERVAL)


def _fault_days(path: Path) -> set[pd.Timestamp]:
    log = pd.read_csv(path, encoding="utf-8-sig")
    log.columns = log.columns.str.strip()
    normalized = {
        column: "".join(char for char in column.lower() if char.isalnum()) for column in log
    }
    date_names = {"date", "datelocal", "localdate", "faultdate", "periodstart"}
    date_column = next(
        (column for column, name in normalized.items() if name in date_names), None
    )
    if date_column is None:
        raise ValueError(
            f"quality log needs a local date column; found columns {log.columns.tolist()}"
        )

    try:
        dates = pd.to_datetime(
            log[date_column].astype(str).str.slice(0, 10), format="%Y-%m-%d", errors="raise"
        ).dt.normalize()
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"quality log column '{date_column}' must contain ISO local dates"
        ) from error

    flag_names = {
        "faulty",
        "isfaulty",
        "fault",
        "meterfault",
        "event",
        "issue",
        "status",
        "qualityflag",
    }
    flag_column = next(
        (column for column, name in normalized.items() if name in flag_names), None
    )
    if flag_column is None:
        return set(dates)

    flagged = []
    for row, value in enumerate(log[flag_column]):
        text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if pd.isna(value) or text in {"", "false", "no", "0", "ok", "reliable", "none"}:
            flagged.append(False)
        elif text in {"true", "yes", "1"} or any(
            marker in text for marker in ("fault", "unreliable", "bad")
        ):
            flagged.append(True)
        else:
            raise ValueError(
                f"quality log column '{flag_column}': row {row} has unknown flag {value!r}"
            )
    return set(dates[pd.Series(flagged, index=dates.index)])


def load_irish(path: str | Path) -> pd.DataFrame:
    """Read Irish UTC interval-start meters and return Dublin-local daily contract rows.

    ``mi_a`` is the grid-incomer consumption meter; ``mi_b`` is same-day PV production and is
    never added to consumption. The quality log is expected beside the dataset's ``extracted``
    directory under ``metadata``; when it has no fault flag column, each listed date is flagged.
    """
    meter_path, quality_path = _resolve_paths(path)
    raw = pd.read_csv(meter_path, encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()
    required = [_TIME_COLUMN, _CONSUMPTION_COLUMN, _PRODUCTION_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"missing Irish column(s) {missing}; found columns {raw.columns.tolist()}")
    if raw.columns.duplicated().any():
        raise ValueError("Irish CSV contains duplicate column names after trimming whitespace")

    try:
        timestamps_utc = pd.to_datetime(
            raw[_TIME_COLUMN], format="ISO8601", utc=True, errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"column '{_TIME_COLUMN}' must contain ISO UTC timestamps") from error
    if timestamps_utc.duplicated().any():
        row = int(timestamps_utc.duplicated().to_numpy().argmax())
        raise ValueError(f"column '{_TIME_COLUMN}': row {row} has a duplicate timestamp")
    if not timestamps_utc.dt.minute.isin([0, 15, 30, 45]).all():
        raise ValueError(f"column '{_TIME_COLUMN}' contains a timestamp off the 15-minute grid")

    consumption = _numeric_column(raw, _CONSUMPTION_COLUMN)
    production = _numeric_column(raw, _PRODUCTION_COLUMN)
    local_days = timestamps_utc.dt.tz_convert(_TIMEZONE).dt.normalize()
    raw = pd.DataFrame(
        {
            "window_start_utc": timestamps_utc,
            "local_day": local_days,
            "mi_a": consumption,
            "mi_b": production,
        }
    )
    daily_groups = list(raw.groupby("local_day", sort=True))
    complete_days = [
        local_day
        for local_day, rows in daily_groups
        if rows["window_start_utc"].nunique() == _expected_intervals(local_day)
    ]
    if not complete_days:
        raise ValueError("Irish meter data contains no complete local day")

    first_complete_day = complete_days[0]
    fault_days = _fault_days(quality_path)
    records = []
    for local_day, rows in daily_groups:
        if local_day < first_complete_day:
            continue
        expected = _expected_intervals(local_day)
        if len(rows) > expected:
            raise ValueError(
                f"column '{_TIME_COLUMN}': {local_day.date()} has more than {expected} intervals"
            )
        kwh = rows["mi_a"].sum(min_count=1)
        if pd.isna(kwh):
            raise ValueError(
                f"column '{_CONSUMPTION_COLUMN}': no valid readings on {local_day.date()}"
            )
        period = local_day.tz_localize(None)
        records.append(
            {
                "period_start": period,
                "period_end": period,
                "kwh": float(kwh),
                "production": float(rows["mi_b"].sum(min_count=1)),
                "production_unit": "proxy:pv_kwh",
                "operating_days": 1.0,
                "source": "real_meter",
                "dataset": "irish_mi_a",
                "coverage_pct": float(rows["mi_a"].count()) / expected * 100,
                "event": "meter_fault" if period in fault_days else None,
            }
        )
    return validate(pd.DataFrame.from_records(records))
