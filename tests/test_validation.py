"""Tests for saveproof/validation.py: windows, injection, run_window and the pre-registered
criteria of docs/validation_plan.md §4-§5 (I2 as amended in §8).

All frames and results tables are small, synthetic and built here; rng = np.random.default_rng(42).
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from saveproof.engine.baseline import add_calendar_terms, usable_rows
from saveproof.engine.savings import SavingsResult
from saveproof.validation import (
    SPECS,
    fixed_split,
    i1,
    i2_amended,
    i2_original,
    i3,
    inject_saving,
    n1,
    n2,
    rolling_windows,
    run_window,
)

FORMULA_B = "kwh ~ production + C(day_type)"
FORMULA_A = "kwh ~ C(day_type)"


def daily_frame(start="2018-01-01", n=365, dates=None, rng=None, dataset="test"):
    """Daily contract table: kwh = 180 + 0.4*production - 30*weekend + noise (CV(RMSE) ~ 2%)."""
    rng = rng if rng is not None else np.random.default_rng(42)
    dates = pd.DatetimeIndex(dates) if dates is not None else pd.date_range(start, periods=n)
    n = len(dates)
    production = rng.uniform(100, 300, n)
    weekend = (dates.dayofweek >= 5).astype(float)
    return pd.DataFrame(
        {
            "period_start": dates,
            "period_end": dates,
            "kwh": 180 + 0.4 * production - 30 * weekend + rng.normal(0, 5.0, n),
            "production": production,
            "production_unit": "t",
            "operating_days": 1.0,
            "source": "simulated",
            "dataset": dataset,
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


def day(text):
    return pd.Timestamp(text)


# ---------- SPECS (validation_plan §3) ----------


def test_specs_are_the_four_preregistered_formulas():
    assert SPECS == {
        "UCI-A": ("uci_steel", "kwh ~ C(day_type)"),
        "UCI-B": ("uci_steel", "kwh ~ production + C(day_type)"),
        "IR-A": ("irish_mi_a", "kwh ~ C(day_type)"),
        "IR-B": ("irish_mi_a", "kwh ~ production + C(day_type)"),
    }


# ---------- fixed_split (validation_plan §4) ----------


def test_uci_fixed_split_is_jan_aug_vs_sep_dec():
    """2018 Jan-Aug = 31+28+31+30+31+30+31+31 = 243 days; Sep-Dec = 30+31+30+31 = 122."""
    window = fixed_split(daily_frame(dataset="uci_steel"), "uci_steel")
    assert window.name == "fixed"
    assert (window.baseline_start, window.baseline_end) == (day("2018-01-01"), day("2018-08-31"))
    assert (window.reporting_start, window.reporting_end) == (day("2018-09-01"), day("2018-12-31"))
    assert len(window.baseline) == 243
    assert len(window.reporting) == 122
    assert window.baseline["period_start"].max() == day("2018-08-31")
    assert window.reporting["period_start"].min() == day("2018-09-01")


def test_irish_fixed_split_is_first_60_percent_of_rows():
    """10 rows -> k = round(6.0) = 6 baseline rows; 11 rows -> round(6.6) = 7."""
    frame = daily_frame(start="2025-05-23", n=10, dataset="irish_mi_a")
    window = fixed_split(frame, "irish_mi_a")
    assert len(window.baseline) == 6
    assert len(window.reporting) == 4
    assert window.baseline_end == day("2025-05-28")
    assert window.reporting_start == day("2025-05-29")
    assert window.reporting_end == day("2025-06-01")
    frame = daily_frame(start="2025-05-23", n=11, dataset="irish_mi_a")
    assert len(fixed_split(frame, "irish_mi_a").baseline) == 7


def test_irish_fixed_split_counts_rows_not_calendar_days():
    """With one date missing, the split is by rows: 9 rows -> round(5.4) = 5 baseline rows."""
    dates = pd.date_range("2025-08-13", periods=10).delete(3)  # drop 16 Aug
    window = fixed_split(daily_frame(dates=dates, dataset="irish_mi_a"), "irish_mi_a")
    assert len(window.baseline) == 5
    assert window.baseline_end == day("2025-08-18")


def test_fixed_split_rejects_unknown_or_mismatched_dataset():
    with pytest.raises(ValueError, match="dataset"):
        fixed_split(daily_frame(dataset="uci_steel"), "other")
    with pytest.raises(ValueError, match="dataset"):
        fixed_split(daily_frame(dataset="uci_steel"), "irish_mi_a")


# ---------- rolling_windows (validation_plan §4) ----------


def test_rolling_365_days_180_90_gives_14_windows_with_calendar_bounds():
    """Window i starts on day 7i; it fits while 7i + 270 <= 365, i.e. i = 0..13.
    Window 0: baseline 1 Jan..29 Jun (day of year 1..180), reporting 30 Jun..27 Sep (181..270).
    Window 13: baseline from 2 Apr (doy 92); reporting ends 27 Dec (doy 361)."""
    windows = rolling_windows(daily_frame(), 180, 90)
    assert len(windows) == 14
    first, last = windows[0], windows[-1]
    assert (first.baseline_start, first.baseline_end) == (day("2018-01-01"), day("2018-06-29"))
    assert (first.reporting_start, first.reporting_end) == (day("2018-06-30"), day("2018-09-27"))
    assert last.baseline_start == day("2018-04-02")
    assert last.reporting_end == day("2018-12-27")
    for i, window in enumerate(windows):
        assert window.baseline_start == day("2018-01-01") + pd.Timedelta(days=7 * i)
        assert len(window.baseline) == 180
        assert len(window.reporting) == 90
        assert window.baseline["period_start"].min() == window.baseline_start
        assert window.baseline["period_start"].max() == window.baseline_end
        assert window.reporting["period_start"].min() == window.reporting_start
        assert window.reporting["period_start"].max() == window.reporting_end
    assert len({w.name for w in windows}) == 14


def test_rolling_keeps_only_windows_that_fit():
    """270 days hold exactly one 180+90 window; 276 still one; 277 two."""
    assert len(rolling_windows(daily_frame(n=269), 180, 90)) == 0
    assert len(rolling_windows(daily_frame(n=270), 180, 90)) == 1
    assert len(rolling_windows(daily_frame(n=276), 180, 90)) == 1
    assert len(rolling_windows(daily_frame(n=277), 180, 90)) == 2


def test_rolling_uses_calendar_days_when_a_date_is_missing():
    """184 calendar days (183 rows, 16 Aug missing), 90/45 step 7: 7i + 135 <= 184 -> 8 windows.
    Counting rows instead (183) would give 7i + 135 <= 183 -> 7 windows."""
    dates = pd.date_range("2025-05-23", periods=184)
    dates = dates.delete(dates.get_loc(day("2025-08-16")))
    windows = rolling_windows(daily_frame(dates=dates), 90, 45)
    assert len(windows) == 8
    assert windows[-1].reporting_end == day("2025-11-22")  # last date in the frame
    for window in windows:
        assert window.baseline_end - window.baseline_start == pd.Timedelta(days=89)
        assert window.reporting_end - window.reporting_start == pd.Timedelta(days=44)
        gap_in_baseline = window.baseline_start <= day("2025-08-16") <= window.baseline_end
        gap_in_reporting = window.reporting_start <= day("2025-08-16") <= window.reporting_end
        assert len(window.baseline) == 90 - gap_in_baseline
        assert len(window.reporting) == 45 - gap_in_reporting
    # window 0: baseline 23 May..20 Aug holds the gap
    assert len(windows[0].baseline) == 89


def test_window_is_frozen():
    window = rolling_windows(daily_frame(), 180, 90)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        window.name = "x"


# ---------- inject_saving (validation_plan §5) ----------


def awkward_reporting():
    """14 days from Mon 1 Jan 2018 with three unusable rows whose kWh must not set the factor:
    a shutdown weekday at 1e6 kWh, a low-coverage weekend day, and a weekday with NaN production
    (unusable for FORMULA_B only)."""
    frame = daily_frame(n=14)
    frame.loc[2, ["event", "kwh"]] = ["shutdown", 1e6]  # Wed
    frame.loc[5, "coverage_pct"] = 50.0  # Sat
    frame.loc[8, "production"] = np.nan  # Tue
    return frame


def usable_total_kwh(frame, formula):
    return frame.loc[usable_rows(add_calendar_terms(frame), formula), "kwh"].sum()


@pytest.mark.parametrize("formula", [FORMULA_A, FORMULA_B])
@pytest.mark.parametrize("shape", ["uniform", "weekday_only"])
@pytest.mark.parametrize("s", [0.0, 0.05, 0.10, 0.15])
def test_injection_drops_usable_total_by_exactly_s(formula, shape, s):
    reporting = awkward_reporting()
    injected = inject_saving(reporting, formula, s, shape)
    # rel 1e-12: one multiplication per row, so only float rounding separates the totals
    assert usable_total_kwh(injected, formula) == pytest.approx(
        (1 - s) * usable_total_kwh(reporting, formula), rel=1e-12
    )


def test_uniform_scales_every_row():
    reporting = awkward_reporting()
    injected = inject_saving(reporting, FORMULA_B, 0.12, "uniform")
    np.testing.assert_allclose(injected["kwh"], 0.88 * reporting["kwh"], rtol=1e-15)


@pytest.mark.parametrize("formula", [FORMULA_A, FORMULA_B])
def test_weekday_only_leaves_weekends_untouched_and_scales_weekdays_by_one_factor(formula):
    reporting = awkward_reporting()
    injected = inject_saving(reporting, formula, 0.10, "weekday_only")
    weekend = reporting["period_start"].dt.dayofweek >= 5
    pd.testing.assert_series_equal(injected.loc[weekend, "kwh"], reporting.loc[weekend, "kwh"])
    factors = injected.loc[~weekend, "kwh"] / reporting.loc[~weekend, "kwh"]
    assert factors.max() - factors.min() < 1e-12
    assert factors.iloc[0] < 0.9  # weekdays cut harder than s so the total drops by s


def test_weekday_only_factor_depends_on_which_rows_the_formula_can_use():
    """Row 8 (weekday, NaN production) counts for FORMULA_A but not FORMULA_B."""
    reporting = awkward_reporting()
    factor_a = (
        inject_saving(reporting, FORMULA_A, 0.1, "weekday_only")["kwh"][0] / reporting["kwh"][0]
    )
    factor_b = (
        inject_saving(reporting, FORMULA_B, 0.1, "weekday_only")["kwh"][0] / reporting["kwh"][0]
    )
    assert factor_a != pytest.approx(factor_b, rel=1e-9)


def test_injection_returns_a_copy_and_touches_only_kwh():
    reporting = awkward_reporting()
    before = reporting.copy()
    injected = inject_saving(reporting, FORMULA_B, 0.15, "weekday_only")
    pd.testing.assert_frame_equal(reporting, before)
    pd.testing.assert_frame_equal(injected.drop(columns="kwh"), before.drop(columns="kwh"))


def test_injection_rejects_bad_arguments():
    reporting = awkward_reporting()
    with pytest.raises(ValueError, match="shape"):
        inject_saving(reporting, FORMULA_B, 0.1, "high_load")
    with pytest.raises(ValueError, match="s "):
        inject_saving(reporting, FORMULA_B, 1.0, "uniform")
    with pytest.raises(ValueError, match="s "):
        inject_saving(reporting, FORMULA_B, -0.05, "uniform")
    weekends_only = reporting[reporting["period_start"].dt.dayofweek >= 5]
    with pytest.raises(ValueError, match="weekday"):
        inject_saving(weekends_only, FORMULA_B, 0.1, "weekday_only")


def test_weekday_only_raises_when_weekdays_cannot_carry_the_cut():
    """Weekdays hold ~1% of the usable total, so a 10% cut would need a negative factor."""
    reporting = daily_frame(n=14)
    weekend = reporting["period_start"].dt.dayofweek >= 5
    reporting.loc[weekend, "kwh"] = 1e6
    with pytest.raises(ValueError, match="weekday"):
        inject_saving(reporting, FORMULA_B, 0.1, "weekday_only")


# ---------- run_window ----------

EXTRA_KEYS = {"cvrmse", "nmbe", "r2", "rho", "n_eff", "p", "contains_zero", "status"}


def test_run_window_returns_savings_fields_model_stats_and_status():
    frame = daily_frame()
    result = run_window(frame.iloc[:180], frame.iloc[180:270], FORMULA_B)
    savings_fields = {f.name for f in dataclasses.fields(SavingsResult)}
    assert savings_fields | EXTRA_KEYS <= set(result)
    assert result["status"] == "ok"
    assert result["p"] == 3
    assert result["m"] == 90
    assert result["contains_zero"] == (result["band_low_pct"] <= 0 <= result["band_high_pct"])
    assert result["contains_zero"]  # nothing changed between the periods
    assert result["verdict"] != "verified"
    assert isinstance(result["gate_failures"], str)


def test_run_window_reports_insufficient_data_instead_of_raising():
    frame = daily_frame()
    baseline = frame.iloc[:180].copy()
    baseline.loc[:59, "event"] = "shutdown"  # 60 of 180 = 33% > MAX_EXCLUDED_SHARE
    result = run_window(baseline, frame.iloc[180:270], FORMULA_B)
    assert result["status"] == "insufficient_data"
    assert result["verdict"] is None
    assert result["contains_zero"] is None
    assert "excluded" in result["status_detail"]
    savings_fields = {f.name for f in dataclasses.fields(SavingsResult)}
    assert savings_fields | EXTRA_KEYS <= set(result)


def test_run_window_insufficient_reporting_data_also_reported():
    frame = daily_frame()
    reporting = frame.iloc[180:270].copy()
    reporting.loc[:, "coverage_pct"] = 10.0
    assert run_window(frame.iloc[:180], reporting, FORMULA_B)["status"] == "insufficient_data"


def test_run_window_lets_other_errors_propagate():
    frame = daily_frame()
    reporting = frame.iloc[180:270].copy()
    reporting.loc[200, "kwh"] = -1.0
    with pytest.raises(ValueError, match="kwh"):
        run_window(frame.iloc[:180], reporting, FORMULA_B)


# ---------- criteria (validation_plan §4, §5, §8) ----------


def null_table(rows):
    """rows: (spec, status, verdict, contains_zero)."""
    return pd.DataFrame(rows, columns=["spec", "status", "verdict", "contains_zero"])


def injected_table(rows):
    """rows: (spec, shape, injected_saving_pct, status, verdict, ashrae_ok)."""
    return pd.DataFrame(
        rows,
        columns=["spec", "shape", "injected_saving_pct", "status", "verdict", "ashrae_ok"],
    )


INSUFFICIENT = ("insufficient_data", None, None)


def test_n1_passes_with_no_verified_and_counts_insufficient():
    table = null_table(
        [
            ("UCI-A", "ok", "likely_not_proven", True),
            ("UCI-B", "ok", "below", False),
            ("IR-B", *INSUFFICIENT),
        ]
    )
    assert n1(table) == {"pass": True, "n_verified": 0, "n_scored": 2, "n_insufficient": 1}


def test_n1_fails_on_one_verified():
    table = null_table(
        [("UCI-B", "ok", "verified", False), ("UCI-B", "ok", "likely_not_proven", True)]
    )
    result = n1(table)
    assert result["pass"] is False
    assert result["n_verified"] == 1


def test_criteria_reject_an_unknown_status():
    with pytest.raises(ValueError, match="status"):
        n1(null_table([("UCI-B", "error", None, None)]))


def test_n2_share_per_spec_and_flags_adequate_spec_below_half():
    """UCI-B: 1 of 3 scored contain 0 (33%) -> flagged. IR-B: 2 of 2 -> not flagged.
    UCI-A: 0 of 1 -> not flagged (calendar spec, not an adequate spec)."""
    table = null_table(
        [
            ("UCI-B", "ok", "likely_not_proven", True),
            ("UCI-B", "ok", "below", False),
            ("UCI-B", "ok", "below", False),
            ("UCI-B", *INSUFFICIENT),
            ("IR-B", "ok", "likely_not_proven", True),
            ("IR-B", "ok", "likely_not_proven", True),
            ("UCI-A", "ok", "likely_not_proven", False),
        ]
    )
    result = n2(table)
    assert result["UCI-B"] == {
        "share_contains_zero": pytest.approx(1 / 3),
        "n_scored": 3,
        "n_insufficient": 1,
        "flagged": True,
    }
    assert result["IR-B"]["share_contains_zero"] == 1.0
    assert result["IR-B"]["flagged"] is False
    assert result["UCI-A"]["share_contains_zero"] == 0.0
    assert result["UCI-A"]["flagged"] is False


def test_n2_exactly_half_is_not_flagged_and_no_scored_windows_gives_none():
    table = null_table(
        [
            ("UCI-B", "ok", "below", True),
            ("UCI-B", "ok", "below", False),
            ("IR-B", *INSUFFICIENT),
        ]
    )
    result = n2(table)
    assert result["UCI-B"]["flagged"] is False
    assert result["IR-B"]["share_contains_zero"] is None
    assert result["IR-B"]["n_insufficient"] == 1


def test_i1_passes_when_no_verified_at_0_5_7_percent():
    table = injected_table(
        [
            ("UCI-B", "uniform", 0, "ok", "below", True),
            ("UCI-B", "weekday_only", 7, "ok", "likely_not_proven", True),
            ("IR-A", "uniform", 5, *INSUFFICIENT),
            ("UCI-B", "uniform", 10, "ok", "verified", True),  # 10% is outside I1
        ]
    )
    assert i1(table) == {"pass": True, "n_verified": 0, "n_scored": 2, "n_insufficient": 1}


def test_i1_fails_on_verified_at_7_percent():
    table = injected_table(
        [
            ("IR-B", "weekday_only", 7, "ok", "verified", True),
            ("IR-B", "uniform", 7, "ok", "likely_not_proven", True),
        ]
    )
    result = i1(table)
    assert result["pass"] is False
    assert result["n_verified"] == 1


def test_i2_amended_fails_only_when_a_gate_failed_window_verifies():
    passing = injected_table(
        [
            ("IR-A", "uniform", 15, "ok", "verified", True),  # gate passed: judged by I1/I3
            ("UCI-A", "uniform", 15, "ok", "likely_not_proven", False),
            ("UCI-A", "uniform", 15, *INSUFFICIENT),
        ]
    )
    assert i2_amended(passing) == {
        "pass": True,
        "n_verified": 0,
        "n_scored": 1,
        "n_insufficient": 1,
    }
    failing = injected_table([("UCI-A", "uniform", 12, "ok", "verified", False)])
    assert i2_amended(failing)["pass"] is False


def test_i2_original_requires_no_verified_for_calendar_specs_at_any_s():
    table = injected_table(
        [
            ("IR-A", "uniform", 15, "ok", "verified", True),
            ("UCI-A", "uniform", 15, "ok", "likely_not_proven", False),
            ("UCI-B", "uniform", 15, "ok", "verified", True),  # not a calendar spec
        ]
    )
    result = i2_original(table)
    assert result == {"pass": False, "n_verified": 1, "n_scored": 2, "n_insufficient": 0}
    assert i2_original(table[table["spec"] != "IR-A"])["pass"] is True


def test_i3_verified_rate_per_spec_shape_and_s():
    table = injected_table(
        [
            ("UCI-B", "uniform", 10, "ok", "verified", True),
            ("UCI-B", "uniform", 10, "ok", "likely_not_proven", True),
            ("UCI-B", "uniform", 10, *INSUFFICIENT),
            ("UCI-B", "uniform", 15, "ok", "verified", True),
            ("UCI-B", "weekday_only", 12, "ok", "likely_not_proven", True),
            ("UCI-B", "uniform", 5, "ok", "verified", True),  # s = 5 is outside I3
        ]
    )
    result = i3(table)
    assert result["UCI-B"]["uniform"]["10"] == {
        "verified_rate": 0.5,
        "n_scored": 2,
        "n_insufficient": 1,
    }
    assert result["UCI-B"]["uniform"]["15"]["verified_rate"] == 1.0
    assert result["UCI-B"]["uniform"]["12"] == {
        "verified_rate": None,
        "n_scored": 0,
        "n_insufficient": 0,
    }
    assert result["UCI-B"]["weekday_only"]["12"]["verified_rate"] == 0.0
    assert "5" not in result["UCI-B"]["uniform"]
