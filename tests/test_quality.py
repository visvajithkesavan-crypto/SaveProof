import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from saveproof.quality import score


def make_quality_df() -> pd.DataFrame:
    days = pd.date_range("2025-01-01", periods=30, freq="D")
    kwh = np.linspace(1000.0, 1040.0, len(days))
    return pd.DataFrame(
        {
            "period_start": days,
            "period_end": days,
            "kwh": kwh,
            "production": kwh / 10,
            "production_unit": ["t"] * len(days),
            "operating_days": [1.0] * len(days),
            "source": ["real_meter"] * len(days),
            "dataset": ["test_plant"] * len(days),
            "coverage_pct": [100.0] * len(days),
            "event": [None] * len(days),
        }
    )


def flag_codes(report) -> set[str]:
    return {flag.code for flag in report.flags}


def test_clean_frame_scores_one_hundred_without_flags():
    report = score(make_quality_df())

    assert report.total == 100
    assert report.completeness == 100
    assert report.continuity == 100
    assert report.plausibility == 100
    assert report.consistency == 100
    assert report.flags == ()


def test_dropped_rows_reduce_continuity_and_name_gap_dates():
    df = make_quality_df().drop(index=[10, 11, 12]).reset_index(drop=True)

    report = score(df)

    assert report.continuity == 90
    gap = next(flag for flag in report.flags if flag.code == "gap")
    assert "2025-01-11" in gap.message
    assert "2025-01-13" in gap.message


def test_zero_coverage_is_a_gap_and_reduces_completeness():
    df = make_quality_df()
    df.loc[10, "coverage_pct"] = 0

    report = score(df)

    assert "gap" in flag_codes(report)
    assert report.completeness < 100


def test_missing_kwh_is_flagged_and_counts_as_zero_completeness():
    df = make_quality_df()
    df.loc[4, "kwh"] = np.nan

    report = score(df)

    assert "missing_kwh" in flag_codes(report)
    assert report.completeness < 100


def test_partial_coverage_is_a_warning():
    df = make_quality_df()
    df.loc[4, "coverage_pct"] = 80

    report = score(df)

    partial = next(flag for flag in report.flags if flag.code == "partial_coverage")
    assert partial.severity == "warning"
    assert report.completeness < 100


def test_negative_kwh_is_flagged_without_modifying_input():
    df = make_quality_df()
    df.loc[4, "kwh"] = -1
    original = df.copy(deep=True)

    report = score(df)

    assert "negative_kwh" in flag_codes(report)
    assert_frame_equal(df, original)


def test_four_consecutive_zero_production_periods_are_suspect():
    df = make_quality_df()
    df.loc[4:7, "production"] = 0

    report = score(df)

    suspect = next(
        flag for flag in report.flags if flag.code == "suspect_zero_production"
    )
    assert suspect.severity == "warning"
    assert "2025-01-05" in suspect.message
    assert "2025-01-08" in suspect.message
    assert "data left unchanged" in suspect.message


def test_two_consecutive_zero_production_periods_are_not_suspect():
    df = make_quality_df()
    df.loc[4:5, "production"] = 0

    assert "suspect_zero_production" not in flag_codes(score(df))


def test_large_kwh_per_production_shift_reduces_consistency():
    df = make_quality_df()
    baseline_end = pd.Timestamp("2025-01-15")
    after_baseline = df["period_start"] > baseline_end
    df.loc[after_baseline, "kwh"] *= 1.3

    report = score(df, baseline_end=baseline_end)

    assert "kwh_per_production_shift" in flag_codes(report)
    assert report.consistency <= 50


def test_small_kwh_per_production_change_is_not_flagged():
    df = make_quality_df()
    baseline_end = pd.Timestamp("2025-01-15")
    after_baseline = df["period_start"] > baseline_end
    df.loc[after_baseline, "kwh"] *= 1.02

    assert "kwh_per_production_shift" not in flag_codes(
        score(df, baseline_end=baseline_end)
    )


def test_proxy_shift_message_mentions_possible_power_factor_change():
    df = make_quality_df()
    df["production_unit"] = "proxy:pv_kwh"
    baseline_end = pd.Timestamp("2025-01-15")
    df.loc[df["period_start"] > baseline_end, "kwh"] *= 1.3

    report = score(df, baseline_end=baseline_end)
    shift = next(
        flag for flag in report.flags if flag.code == "kwh_per_production_shift"
    )

    assert "power-factor change" in shift.message


def test_empty_frame_raises_no_rows():
    with pytest.raises(ValueError, match="^no rows$"):
        score(make_quality_df().iloc[:0])


def test_missing_required_column_raises():
    df = make_quality_df().drop(columns=["kwh"])

    with pytest.raises(ValueError, match="missing required column"):
        score(df)
