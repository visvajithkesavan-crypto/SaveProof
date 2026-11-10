import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from saveproof.contract import validate
from saveproof.ingest.irish import load_irish

FIXTURES = Path(__file__).parents[1] / "fixtures"
# Exact header and column order of data/raw/irish/*_15min_ALLphase.csv.
METER_COLUMNS = [
    "AssetId",
    "Phase",
    "window_start_utc",
    "DateKeyUtc",
    "TimeKeyUtc",
    "WindowStartLocal",
    "HourLocal",
    "DateLocal",
    "Energy_kWh_15m",
    "Demand_kW",
    "AvgPower_kW_15m",
    "Minutes",
    "SecondsObserved",
    "DataCoveragePct",
    "IsReliableWindow",
    "SecondsReliable",
    "ReliableCoveragePct",
]
# (first window_start_utc, number of windows, kWh per window, unreliable windows at the start)
GRID_SEGMENTS = [
    ("2025-03-29 20:00:00", 16, 999.0, 0),  # partial local day before the first complete day
    ("2025-03-30 00:00:00", 92, 1.0, 1),  # spring-forward local day: 92 windows
    ("2025-05-22 23:00:00", 96, 10.0, 0),  # local May 23 starts 23:00 UTC (IST = UTC+1)
    ("2025-05-23 23:00:00", 95, 10.0, 0),  # May 24: last window missing
    ("2025-10-25 23:00:00", 100, 1.0, 0),  # fall-back local day: 100 windows
]
PV_SEGMENTS = [
    ("2025-03-28 00:00:00", 96, 5.0, 0),  # PV day with no grid row: ignored
    ("2025-03-30 00:00:00", 92, 0.5, 0),
    ("2025-05-22 23:00:00", 96, 2.0, 0),
    ("2025-05-23 23:00:00", 96, 2.0, 10),  # 86/96 reliable < MIN_COVERAGE_PCT -> NaN
    # no PV windows on Oct 26 -> NaN
]


def meter_rows(asset_id: str, segments: list[tuple[str, int, float, int]]) -> pd.DataFrame:
    """Build a meter file in the real long format, rows sorted by window_start_utc."""
    frames = []
    for start, count, kwh, unreliable in segments:
        utc = pd.date_range(start, periods=count, freq="15min", tz="UTC")
        local = utc.tz_convert("Europe/Dublin").tz_localize(None)
        reliable = np.r_[np.zeros(unreliable, dtype=int), np.ones(count - unreliable, dtype=int)]
        frames.append(
            pd.DataFrame(
                {
                    "AssetId": asset_id,
                    "Phase": "ALL",
                    "window_start_utc": utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "DateKeyUtc": utc.strftime("%Y%m%d").astype(int),
                    "TimeKeyUtc": utc.strftime("%H%M%S").astype(int),
                    "WindowStartLocal": local.strftime("%Y-%m-%d %H:%M:%S"),
                    "HourLocal": local.hour,
                    "DateLocal": local.strftime("%Y-%m-%d"),
                    "Energy_kWh_15m": kwh,
                    "Demand_kW": kwh * 4,
                    "AvgPower_kW_15m": kwh * 4,
                    "Minutes": 15.0,
                    "SecondsObserved": 900.0,
                    "DataCoveragePct": 100.0,
                    "IsReliableWindow": reliable,
                    "SecondsReliable": 900.0 * reliable,
                    "ReliableCoveragePct": 100.0 * reliable,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[METER_COLUMNS]


def write_irish_dir(root: Path, grid: pd.DataFrame, pv: pd.DataFrame) -> Path:
    (root / "metadata").mkdir(parents=True)
    shutil.copy(FIXTURES / "meter_data_quality_log.csv", root / "metadata")
    grid.to_csv(root / "mi_a_grid_incomer_15min_ALLphase.csv", index=False)
    pv.to_csv(root / "mi_b_pv_generation_15min_ALLphase.csv", index=False)
    return root


@pytest.fixture
def irish_dir(tmp_path: Path) -> Path:
    grid = meter_rows("mi_a", GRID_SEGMENTS)
    pv = meter_rows("mi_b", PV_SEGMENTS)
    return write_irish_dir(tmp_path / "irish", grid, pv)


def test_irish_daily_rows_use_dublin_days_from_first_complete_grid_day(irish_dir):
    daily = load_irish(irish_dir)
    validate(daily)

    dates = daily["period_start"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2025-03-30", "2025-05-23", "2025-05-24", "2025-10-26"]
    assert daily["kwh"].tolist() == pytest.approx([92.0, 960.0, 950.0, 100.0])
    assert (daily["period_end"] == daily["period_start"]).all()


def test_irish_coverage_counts_reliable_grid_windows_against_dst_day_length(irish_dir):
    daily = load_irish(irish_dir)

    assert daily["coverage_pct"].tolist() == pytest.approx(
        [91 / 92 * 100, 100.0, 95 / 96 * 100, 100.0]
    )


def test_irish_production_is_pv_kwh_or_nan_when_pv_coverage_is_low(irish_dir):
    daily = load_irish(irish_dir)

    assert daily.loc[0, "production"] == pytest.approx(46.0)
    assert daily.loc[1, "production"] == pytest.approx(192.0)
    assert np.isnan(daily.loc[2, "production"])
    assert np.isnan(daily.loc[3, "production"])


def test_irish_meter_fault_only_from_mi_a_and_mi_b_issue_intervals(irish_dir):
    daily = load_irish(irish_dir)

    assert daily["event"].tolist() == [None, "meter_fault", None, "meter_fault"]


def test_irish_contract_metadata(irish_dir):
    daily = load_irish(irish_dir)

    assert daily["production_unit"].eq("proxy:pv_kwh").all()
    assert daily["source"].eq("real_meter").all()
    assert daily["dataset"].eq("irish_mi_a").all()
    assert daily["operating_days"].eq(1.0).all()


def test_irish_raises_when_date_local_disagrees_with_utc_conversion(tmp_path):
    grid = meter_rows("mi_a", GRID_SEGMENTS)
    grid.loc[20, "DateLocal"] = "2025-03-29"
    root = write_irish_dir(tmp_path / "irish", grid, meter_rows("mi_b", PV_SEGMENTS))

    with pytest.raises(ValueError, match="column 'DateLocal': row 20"):
        load_irish(root)


def test_irish_raises_naming_a_missing_column(tmp_path):
    grid = meter_rows("mi_a", GRID_SEGMENTS).drop(columns="IsReliableWindow")
    root = write_irish_dir(tmp_path / "irish", grid, meter_rows("mi_b", PV_SEGMENTS))

    with pytest.raises(ValueError, match="IsReliableWindow"):
        load_irish(root)
