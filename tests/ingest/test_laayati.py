"""Tests for ingest/laayati.py (docs/validation_addendum_v5b.md §3-§4).

The fixture copies the real header "Time,THE,HM ,KWh" (space after HM) and the d/m/yy H:MM
timestamps. Day 1 (31/12/18): KWh = 500 + 10h, THE = 3 from 06:00. Day 2 (1/1/19): KWh = 600,
THE = h, HM blank at 08:00 (as in the real Destoning file). Lines: 0 header, 1-24 day 1, 25-48
day 2.
"""

from pathlib import Path

import pandas as pd
import pytest

from saveproof.contract import validate
from saveproof.ingest.laayati import load_laayati

FIXTURE = Path(__file__).parents[1] / "fixtures" / "laayati_two_days.csv"


def edited(tmp_path, edit):
    """Write the fixture with edit(lines) applied and return the new path."""
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    edit(lines)
    path = tmp_path / "laayati.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_daily_sums_dates_and_coverage():
    """Day 1: kWh = 24*500 + 10*(0+...+23) = 12000 + 2760 = 14760; THE = 18 * 3 = 54.
    Day 2: kWh = 24*600 = 14400; THE = 0+...+23 = 276."""
    daily = load_laayati(FIXTURE, "destoning")
    assert list(daily["period_start"]) == list(pd.to_datetime(["2018-12-31", "2019-01-01"]))
    assert (daily["period_end"] == daily["period_start"]).all()
    assert daily["kwh"].tolist() == [14760.0, 14400.0]
    assert daily["production"].tolist() == [54.0, 276.0]
    assert daily["coverage_pct"].tolist() == [100.0, 100.0]


def test_contract_metadata_is_exact_and_output_passes_validate():
    daily = load_laayati(FIXTURE, "screening")
    pd.testing.assert_frame_equal(validate(daily), daily)
    assert daily["production_unit"].tolist() == ["process_kpi:THE"] * 2
    assert daily["operating_days"].tolist() == [1.0, 1.0]
    assert daily["source"].tolist() == ["real_meter"] * 2
    assert daily["dataset"].tolist() == ["laayati_screening"] * 2
    assert daily["event"].isna().all()


@pytest.mark.parametrize("station", ["destoning", "screening", "loading"])
def test_dataset_names_the_station(station):
    assert set(load_laayati(FIXTURE, station)["dataset"]) == {f"laayati_{station}"}


def test_unknown_station_raises():
    with pytest.raises(ValueError, match="station"):
        load_laayati(FIXTURE, "crushing")


def test_day_with_23_hours_raises(tmp_path):
    path = edited(tmp_path, lambda lines: lines.pop(30))  # drop 1/1/19 5:00
    with pytest.raises(ValueError, match=r"2019-01-01 has 23 hours, expected 24"):
        load_laayati(path, "destoning")


def test_day_with_25_hours_raises(tmp_path):
    """An extra 1/1/19 4:00 row gives day 2 25 rows."""
    path = edited(tmp_path, lambda lines: lines.insert(30, "1/1/19 4:00,4,0.5,600"))
    with pytest.raises(ValueError, match=r"2019-01-01 has 25 hours, expected 24"):
        load_laayati(path, "destoning")


def test_trailing_partial_day_raises(tmp_path):
    def add_two_hours(lines):
        lines.append("2/1/19 0:00,0,0,600")
        lines.append("2/1/19 1:00,0,0,600")

    with pytest.raises(ValueError, match=r"2019-01-02 has 2 hours, expected 24"):
        load_laayati(edited(tmp_path, add_two_hours), "destoning")


def test_duplicate_hour_raises_even_when_the_day_count_would_hide_it(tmp_path):
    """Replacing 1/1/19 5:00 with a second 4:00 keeps 24 rows on the day."""
    path = edited(tmp_path, lambda lines: lines.__setitem__(30, "1/1/19 4:00,5,0.5,600"))
    with pytest.raises(ValueError, match="duplicate"):
        load_laayati(path, "destoning")


def test_off_the_hour_timestamp_raises(tmp_path):
    path = edited(tmp_path, lambda lines: lines.__setitem__(30, "1/1/19 5:30,5,0.5,600"))
    with pytest.raises(ValueError, match="on the hour"):
        load_laayati(path, "destoning")


def test_month_first_timestamp_is_rejected_not_guessed(tmp_path):
    """13/1/19 is valid d/m/yy; 1/13/19 (m/d/yy) is not and must not be parsed."""
    path = edited(tmp_path, lambda lines: lines.__setitem__(25, "1/13/19 0:00,0,0.5,600"))
    with pytest.raises(ValueError, match="Time"):
        load_laayati(path, "destoning")


def test_missing_kwh_hour_lowers_coverage_and_kwh_sums_the_rest(tmp_path):
    """Day 2 loses one KWh reading: coverage 23/24, kWh = 23 * 600."""
    path = edited(tmp_path, lambda lines: lines.__setitem__(30, "1/1/19 5:00,5,0.5,"))
    daily = load_laayati(path, "destoning")
    assert daily.loc[1, "coverage_pct"] == pytest.approx(100 * 23 / 24)
    assert daily.loc[1, "kwh"] == 23 * 600.0


def test_missing_the_hour_makes_the_day_production_nan_not_a_partial_sum(tmp_path):
    path = edited(tmp_path, lambda lines: lines.__setitem__(30, "1/1/19 5:00,,0.5,600"))
    daily = load_laayati(path, "destoning")
    assert pd.isna(daily.loc[1, "production"])
    assert daily.loc[0, "production"] == 54.0


def test_day_with_no_kwh_readings_raises(tmp_path):
    def blank_day_2(lines):
        for i in range(25, 49):
            lines[i] = lines[i].rsplit(",", 1)[0] + ","

    with pytest.raises(ValueError, match=r"KWh.*2019-01-01"):
        load_laayati(edited(tmp_path, blank_day_2), "destoning")


def test_missing_column_raises(tmp_path):
    path = edited(tmp_path, lambda lines: lines.__setitem__(0, "Time,TH,HM ,KWh"))
    with pytest.raises(ValueError, match="THE"):
        load_laayati(path, "destoning")


def test_non_numeric_reading_raises(tmp_path):
    path = edited(tmp_path, lambda lines: lines.__setitem__(3, "31/12/18 2:00,0,0,abc"))
    with pytest.raises(ValueError, match="KWh"):
        load_laayati(path, "destoning")


def test_negative_reading_raises(tmp_path):
    path = edited(tmp_path, lambda lines: lines.__setitem__(3, "31/12/18 2:00,-1,0,520"))
    with pytest.raises(ValueError, match="THE"):
        load_laayati(path, "destoning")
