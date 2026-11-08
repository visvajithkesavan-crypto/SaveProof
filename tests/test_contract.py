"""Tests for the data contract: a valid frame passes; each rule raises naming column and row."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from saveproof.contract import REQUIRED, SOURCES, PlantMeta, validate


def make_valid_df() -> pd.DataFrame:
    """Three consecutive daily rows (period_end inclusive, so start == end)."""
    days = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    return pd.DataFrame(
        {
            "period_start": days,
            "period_end": days,
            "kwh": [1000.0, 1100.0, 950.0],
            "production": [10.0, None, 9.5],  # production is nullable
            "production_unit": ["t", "t", "t"],
            "operating_days": [1.0, 1.0, 1.0],
            "source": ["real_meter", "real_meter", "real_meter"],
            "dataset": ["test_plant"] * 3,
            "coverage_pct": [100.0, 100.0, 100.0],
            "event": [None, None, "holiday"],  # event is nullable
        }
    )


def make_meta(**overrides) -> PlantMeta:
    fields = {
        "plant_id": "p1",
        "sector": "foundry",
        "timezone": "Asia/Kolkata",
        "baseline_start": pd.Timestamp("2025-01-01"),
        "baseline_end": pd.Timestamp("2025-06-30"),
        "reporting_start": pd.Timestamp("2025-07-02"),
        "reporting_end": pd.Timestamp("2025-12-31"),
        "retrofit_date": None,
    }
    return PlantMeta(**{**fields, **overrides})


def test_valid_frame_passes_and_is_returned_unchanged():
    df = make_valid_df()
    out = validate(df)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df)


def test_validate_returns_sorted_copy_and_leaves_input_untouched():
    df = make_valid_df().iloc[::-1].reset_index(drop=True)  # newest first
    before = df.copy()
    out = validate(df)
    assert list(out["period_start"]) == sorted(df["period_start"])
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("column", REQUIRED)
def test_missing_required_column_raises_naming_column(column):
    df = make_valid_df().drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        validate(df)


def test_bad_source_raises_naming_column_and_value():
    df = make_valid_df()
    df.loc[1, "source"] = "csv_upload"
    with pytest.raises(ValueError, match="source") as excinfo:
        validate(df)
    assert "csv_upload" in str(excinfo.value)


def test_sources_are_the_three_contract_values():
    assert {"real_meter", "utility_bill", "simulated"} == SOURCES


def test_negative_kwh_raises_naming_column():
    df = make_valid_df()
    df.loc[2, "kwh"] = -1.0
    with pytest.raises(ValueError, match="kwh"):
        validate(df)


def test_period_end_before_period_start_raises_naming_column():
    df = make_valid_df()
    df.loc[0, "period_end"] = pd.Timestamp("2024-12-31")
    with pytest.raises(ValueError, match="period_end"):
        validate(df)


def test_single_day_period_with_inclusive_end_is_valid():
    df = make_valid_df()
    assert (df["period_start"] == df["period_end"]).all()
    validate(df)


def test_duplicate_periods_raise_naming_column():
    df = make_valid_df()
    df.loc[2, ["period_start", "period_end"]] = [pd.Timestamp("2025-01-01")] * 2
    with pytest.raises(ValueError, match="period_start"):
        validate(df)


# --- 2a: period columns must be datetime64 at midnight -------------------------------------


@pytest.mark.parametrize("column", ["period_start", "period_end"])
@pytest.mark.parametrize("as_text", [True, False])
def test_period_columns_must_be_datetime64_not_object(column, as_text):
    df = make_valid_df()
    df[column] = df[column].dt.strftime("%Y-%m-%d") if as_text else df[column].dt.date
    with pytest.raises(ValueError, match=rf"column '{column}'.*datetime64"):
        validate(df)


@pytest.mark.parametrize("column", ["period_start", "period_end"])
def test_period_columns_must_be_normalized_to_midnight(column):
    df = make_valid_df()
    df.loc[1, column] = pd.Timestamp("2025-01-02 06:00")
    with pytest.raises(ValueError, match=rf"column '{column}'.*row 1.*midnight"):
        validate(df)


# --- 2b: no nulls in the non-nullable columns ----------------------------------------------

NOT_NULL = [
    "period_start",
    "period_end",
    "kwh",
    "operating_days",
    "source",
    "dataset",
    "production_unit",
    "coverage_pct",
]


@pytest.mark.parametrize("column", NOT_NULL)
def test_null_in_required_value_raises_naming_column_and_row(column):
    df = make_valid_df()
    df.loc[1, column] = None
    with pytest.raises(ValueError, match=rf"column '{column}'.*row 1.*null"):
        validate(df)


def test_null_production_and_event_are_allowed():
    df = make_valid_df()
    df["production"] = np.nan  # float NaN, not an object column of None
    df["event"] = None
    validate(df)


# --- 2c-2f: value ranges -------------------------------------------------------------------


def test_negative_production_raises_naming_column_and_row():
    df = make_valid_df()
    df.loc[0, "production"] = -0.5
    with pytest.raises(ValueError, match=r"column 'production'.*row 0"):
        validate(df)


def test_zero_production_is_valid():
    df = make_valid_df()
    df.loc[0, "production"] = 0.0
    validate(df)


@pytest.mark.parametrize("operating_days", [-1.0, 2.0])  # 2 > 1 day in a one-day period
def test_operating_days_outside_period_length_raises(operating_days):
    df = make_valid_df()
    df.loc[2, "operating_days"] = operating_days
    with pytest.raises(ValueError, match=r"column 'operating_days'.*row 2"):
        validate(df)


def test_operating_days_may_equal_inclusive_period_length():
    df = make_valid_df().iloc[:1].copy()
    df["period_end"] = pd.Timestamp("2025-01-07")  # 7 days, end inclusive
    df["operating_days"] = 7.0
    validate(df)
    df["operating_days"] = 7.5
    with pytest.raises(ValueError, match=r"column 'operating_days'.*row 0"):
        validate(df)


@pytest.mark.parametrize("coverage_pct", [-0.1, 100.1])
def test_coverage_pct_outside_0_100_raises(coverage_pct):
    df = make_valid_df()
    df.loc[1, "coverage_pct"] = coverage_pct
    with pytest.raises(ValueError, match=r"column 'coverage_pct'.*row 1"):
        validate(df)


@pytest.mark.parametrize("coverage_pct", [0.0, 100.0])
def test_coverage_pct_bounds_are_valid(coverage_pct):
    df = make_valid_df()
    df["coverage_pct"] = coverage_pct
    validate(df)


def test_events_are_the_four_contract_values():
    from saveproof.contract import EVENTS

    assert {"retrofit", "shutdown", "meter_fault", "holiday"} == EVENTS


def test_unknown_event_raises_naming_column_and_row():
    df = make_valid_df()
    df.loc[1, "event"] = "fire"
    with pytest.raises(ValueError, match=r"column 'event'.*row 1.*fire"):
        validate(df)


@pytest.mark.parametrize("event", ["retrofit", "shutdown", "meter_fault", "holiday"])
def test_each_known_event_is_valid(event):
    df = make_valid_df()
    df["event"] = event
    validate(df)


# --- 2g: one plant at a time ---------------------------------------------------------------


def test_more_than_one_dataset_raises_naming_column_and_row():
    df = make_valid_df()
    df.loc[2, "dataset"] = "other_plant"
    with pytest.raises(ValueError, match=r"column 'dataset'.*row 2.*other_plant"):
        validate(df)


@pytest.mark.parametrize("empty", [make_valid_df().iloc[0:0], pd.DataFrame()], ids=["cols", "bare"])
def test_empty_frame_raises_no_rows(empty):
    with pytest.raises(ValueError, match="no rows"):
        validate(empty)


NUMERIC = ["kwh", "production", "operating_days", "coverage_pct"]


@pytest.mark.parametrize("column", NUMERIC)
def test_numeric_columns_must_have_numeric_dtype(column):
    df = make_valid_df()
    df[column] = ["1", "2", "3"]
    with pytest.raises(ValueError, match=rf"column '{column}'.*numeric"):
        validate(df)


@pytest.mark.parametrize("column", NUMERIC)
@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_numeric_columns_must_be_finite(column, bad):
    df = make_valid_df()
    df.loc[0, column] = bad
    with pytest.raises(ValueError, match=rf"column '{column}'.*row 0.*infinite"):
        validate(df)


def test_more_than_one_production_unit_raises_naming_column_and_row():
    df = make_valid_df()
    df.loc[2, "production_unit"] = "kg"
    with pytest.raises(ValueError, match=r"column 'production_unit'.*row 2.*kg"):
        validate(df)


# --- 2h: no overlapping periods ------------------------------------------------------------


def test_overlapping_periods_raise_naming_column_and_row():
    df = make_valid_df().iloc[:2].copy()
    df.loc[0, "period_end"] = pd.Timestamp("2025-01-02")  # row 0 now spans Jan 1-2
    df.loc[1, "period_start"] = pd.Timestamp("2025-01-02")  # row 1 starts on row 0's last day
    df["operating_days"] = 1.0
    with pytest.raises(ValueError, match=r"column 'period_start'.*row 1.*overlap"):
        validate(df)


def test_adjacent_multi_day_periods_are_valid():
    df = make_valid_df().iloc[:2].copy()
    df.loc[0, ["period_start", "period_end"]] = [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-31"),
    ]
    df.loc[1, ["period_start", "period_end"]] = [
        pd.Timestamp("2025-02-01"),
        pd.Timestamp("2025-02-28"),
    ]
    df["operating_days"] = [26.0, 24.0]
    validate(df)


def test_duplicate_period_is_reported_as_overlap_on_the_later_row():
    df = make_valid_df()
    df.loc[2, ["period_start", "period_end"]] = [pd.Timestamp("2025-01-01")] * 2
    with pytest.raises(ValueError, match=r"column 'period_start'.*row 2.*overlap"):
        validate(df)


# --- PlantMeta -----------------------------------------------------------------------------


def test_plant_meta_is_frozen():
    meta = make_meta()
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.plant_id = "p2"


def test_plant_meta_valid_dates_pass_including_boundaries():
    make_meta(retrofit_date=pd.Timestamp("2025-07-02"))  # retrofit == reporting_start is allowed
    make_meta(baseline_end=pd.Timestamp("2025-01-01"))  # baseline_start == baseline_end is allowed
    make_meta(reporting_end=pd.Timestamp("2025-07-02"))  # reporting_start == reporting_end


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"baseline_end": pd.Timestamp("2024-12-31")}, "baseline_end"),  # < baseline_start
        ({"reporting_start": pd.Timestamp("2025-06-30")}, "reporting_start"),  # == baseline_end
        ({"reporting_start": pd.Timestamp("2025-06-01")}, "reporting_start"),  # < baseline_end
        ({"reporting_end": pd.Timestamp("2025-07-01")}, "reporting_end"),  # < reporting_start
        ({"retrofit_date": pd.Timestamp("2025-06-30")}, "retrofit_date"),  # == baseline_end
        ({"retrofit_date": pd.Timestamp("2025-03-01")}, "retrofit_date"),  # inside baseline
        ({"retrofit_date": pd.Timestamp("2025-07-03")}, "retrofit_date"),  # > reporting_start
    ],
)
def test_plant_meta_bad_dates_raise_naming_field(overrides, field):
    with pytest.raises(ValueError, match=field):
        make_meta(**overrides)


DATE_FIELDS = [
    "baseline_start",
    "baseline_end",
    "reporting_start",
    "reporting_end",
    "retrofit_date",
]


@pytest.mark.parametrize("field", DATE_FIELDS)
@pytest.mark.parametrize(
    "make_bad",
    [
        lambda t: t.date(),  # datetime.date
        lambda t: str(t.date()),  # text
        lambda t: t.tz_localize("Asia/Kolkata"),  # timezone-aware
        lambda t: pd.Timestamp(f"{t:%Y-%m-%d} 06:00"),  # not at midnight
    ],
    ids=["date", "text", "tz_aware", "not_midnight"],
)
def test_plant_meta_date_fields_must_be_naive_midnight_timestamps(field, make_bad):
    # Same calendar day as the valid value, so the order checks alone could never flag it.
    base = {"retrofit_date": pd.Timestamp("2025-07-01")}
    bad = make_bad(make_meta(**base).__dict__[field])
    with pytest.raises(ValueError, match=field):
        make_meta(**{**base, field: bad})


def test_plant_meta_dates_compare_directly_with_frame_columns():
    meta = make_meta()
    in_baseline = make_valid_df()["period_start"] <= meta.baseline_end
    assert in_baseline.dtype == bool
    assert in_baseline.all()
