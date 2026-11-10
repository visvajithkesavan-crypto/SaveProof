"""Tests for engine/savings.py: verified savings, uncertainty band and the ADEETIE verdict
(rule: docs/validation_plan.md §2).

All frames are synthetic and built here with rng = np.random.default_rng(42).
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from saveproof.config import (
    ADEETIE_THRESHOLD,
    CONFIDENCE_ASHRAE,
    CONFIDENCE_REPORT,
    MAX_EXCLUDED_SHARE,
    MAX_EXTRAPOLATED_SHARE,
)
from saveproof.engine import stats
from saveproof.engine.baseline import InsufficientDataError, fit_baseline, predict
from saveproof.engine.savings import SavingsResult, decide_verdict, verify_savings

FORMULA = "kwh ~ production + C(day_type)"
CALENDAR_FORMULA = "kwh ~ C(day_type)"


def daily_frame(rng, start="2024-01-01", n=365, production_t=None, noise_kwh=5.0):
    """Daily contract table: kwh = 180 + 0.4*production - 30*weekend + noise (CV(RMSE) ~ 2%)."""
    dates = pd.date_range(start, periods=n, freq="D")
    if production_t is None:
        production_t = rng.uniform(100, 300, n)
    weekend = (dates.dayofweek >= 5).astype(float)
    return pd.DataFrame(
        {
            "period_start": dates,
            "period_end": dates,
            "kwh": 180 + 0.4 * production_t - 30 * weekend + rng.normal(0, noise_kwh, n),
            "production": production_t,
            "production_unit": "t",
            "operating_days": 1.0,
            "source": "simulated",
            "dataset": "test",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


def monthly_frame(rng, start="2022-01-01", n=24, noise_kwh=20.0):
    """Monthly bills: kwh = 500 + 0.4*production + 30*operating_days + noise."""
    starts = pd.date_range(start, periods=n, freq="MS")
    operating_days = rng.integers(20, 27, n).astype(float)
    production_t = rng.uniform(2000, 6000, n)
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + pd.offsets.MonthEnd(0),
            "kwh": 500 + 0.4 * production_t + 30 * operating_days + rng.normal(0, noise_kwh, n),
            "production": production_t,
            "production_unit": "t",
            "operating_days": operating_days,
            "source": "utility_bill",
            "dataset": "test",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


def good_plant(rng, n_reporting=180):
    """(model, reporting frame) for a low-noise plant with no retrofit."""
    baseline = daily_frame(rng, "2024-01-01", 365)
    reporting = daily_frame(rng, "2025-01-01", n_reporting)
    return fit_baseline(baseline, FORMULA), reporting


def with_saving(reporting, saving_pct):
    return reporting.assign(kwh=reporting["kwh"] * (1 - saving_pct / 100))


# ---------- decide_verdict (validation_plan §2 steps 3-4) ----------


@pytest.mark.parametrize(
    ("low", "high", "ashrae_ok", "extrapolated_share", "expected"),
    [
        (10.0, 20.0, True, 0.0, "verified"),  # band_low exactly 10 -> verified
        (9.99, 20.0, True, 0.0, "likely_not_proven"),
        (-5.0, 10.0, True, 0.0, "likely_not_proven"),  # band_high exactly 10 -> not "below"
        (-5.0, 9.99, True, 0.0, "below"),
        (5.0, 15.0, True, 0.0, "likely_not_proven"),  # band straddles 10
        (15.0, 25.0, False, 0.0, "likely_not_proven"),  # gate failure beats a verified-looking band
        (-5.0, 5.0, False, 0.0, "likely_not_proven"),  # a bad model can never say "below" either
        (15.0, 25.0, True, MAX_EXTRAPOLATED_SHARE + 0.01, "likely_not_proven"),
        (-5.0, 5.0, True, 0.9, "likely_not_proven"),
        (15.0, 25.0, True, MAX_EXTRAPOLATED_SHARE, "verified"),  # exactly at the limit passes
    ],
)
def test_decide_verdict_truth_table(low, high, ashrae_ok, extrapolated_share, expected):
    assert decide_verdict(low, high, ashrae_ok, extrapolated_share) == expected


def test_max_extrapolated_share_is_a_quarter():
    assert MAX_EXTRAPOLATED_SHARE == 0.25  # docs/validation_plan.md §2 step 2


# ---------- verify_savings: invariants and known answers ----------


def test_same_data_as_baseline_and_reporting_gives_zero_savings_and_below():
    """OLS residuals sum to zero in-sample, so savings are 0 up to float error."""
    baseline = daily_frame(np.random.default_rng(42), n=365)
    result = verify_savings(fit_baseline(baseline, FORMULA), baseline)
    assert result.savings_pct == pytest.approx(0.0, abs=1e-8)
    assert result.band_low_pct < 0 < result.band_high_pct
    assert result.ashrae_ok
    assert result.verdict == "below"


def test_known_20_percent_saving_is_recovered_and_verified():
    """Reporting kWh x 0.80 on a CV(RMSE) ~ 2% plant. Estimation noise (mean of 180 daily errors
    plus fitted-intercept error) is ~0.2 pp, so abs=0.75 pp is ~3.5 standard deviations."""
    model, reporting = good_plant(np.random.default_rng(42))
    result = verify_savings(model, with_saving(reporting, 20.0))
    assert result.savings_pct == pytest.approx(20.0, abs=0.75)
    assert result.band_low_pct >= 100 * ADEETIE_THRESHOLD
    assert result.verdict == "verified"
    assert result.gate_failures == ()


def test_band_width_does_not_depend_on_the_saving_and_is_symmetric():
    model, reporting = good_plant(np.random.default_rng(42))
    small = verify_savings(model, with_saving(reporting, 5.0))
    large = verify_savings(model, with_saving(reporting, 15.0))
    width_small = small.band_high_pct - small.band_low_pct
    width_large = large.band_high_pct - large.band_low_pct
    assert width_small > 0
    assert width_small == pytest.approx(width_large, rel=1e-9)
    for result in (small, large):
        midpoint = (result.band_low_pct + result.band_high_pct) / 2
        assert midpoint == pytest.approx(result.savings_pct, rel=1e-9)


def test_scaling_kwh_in_both_periods_leaves_savings_pct_and_band_unchanged():
    rng = np.random.default_rng(42)
    baseline = daily_frame(rng, "2024-01-01", 365)
    reporting = with_saving(daily_frame(rng, "2025-01-01", 180), 12.0)
    k = 3.0
    a = verify_savings(fit_baseline(baseline, FORMULA), reporting)
    b = verify_savings(
        fit_baseline(baseline.assign(kwh=baseline["kwh"] * k), FORMULA),
        reporting.assign(kwh=reporting["kwh"] * k),
    )
    assert b.savings_kwh == pytest.approx(k * a.savings_kwh, rel=1e-9)
    assert b.savings_pct == pytest.approx(a.savings_pct, rel=1e-9)
    assert b.band_low_pct == pytest.approx(a.band_low_pct, rel=1e-9)
    assert b.band_high_pct == pytest.approx(a.band_high_pct, rel=1e-9)
    assert b.fsu == pytest.approx(a.fsu, rel=1e-9)


def test_totals_and_band_come_from_predict_and_stats():
    model, reporting = good_plant(np.random.default_rng(42))
    result = verify_savings(model, with_saving(reporting, 12.0))
    reporting_12 = with_saving(reporting, 12.0)
    adjusted_kwh = predict(model, reporting_12).sum()
    assert result.adjusted_baseline_kwh == pytest.approx(adjusted_kwh, rel=1e-12)
    assert result.actual_kwh == pytest.approx(reporting_12["kwh"].sum(), rel=1e-12)
    assert result.savings_kwh == pytest.approx(adjusted_kwh - reporting_12["kwh"].sum(), rel=1e-9)
    assert result.savings_pct == pytest.approx(100 * result.savings_kwh / adjusted_kwh, rel=1e-12)
    assert (result.n, result.m, result.m_excluded) == (365, 180, 0)
    assert result.confidence == CONFIDENCE_REPORT

    def half_width_kwh(confidence):
        return stats.savings_uncertainty_kwh(
            model.cvrmse, model.n, model.p, model.rho, 180, adjusted_kwh, confidence, model.monthly
        )

    width_pct = 200 * half_width_kwh(CONFIDENCE_REPORT) / adjusted_kwh
    assert result.band_high_pct - result.band_low_pct == pytest.approx(width_pct, rel=1e-9)
    # FSU is evaluated at a 10% saving, not at the measured saving.
    fsu = half_width_kwh(CONFIDENCE_ASHRAE) / (ADEETIE_THRESHOLD * adjusted_kwh)
    assert result.fsu == pytest.approx(fsu, rel=1e-9)


def test_fsu_does_not_depend_on_the_measured_saving():
    model, reporting = good_plant(np.random.default_rng(42))
    none = verify_savings(model, reporting)
    big = verify_savings(model, with_saving(reporting, 30.0))
    assert big.fsu == pytest.approx(none.fsu, rel=1e-12)


def test_confidence_argument_widens_the_band():
    model, reporting = good_plant(np.random.default_rng(42))
    narrow = verify_savings(model, reporting, confidence=0.80)
    wide = verify_savings(model, reporting, confidence=0.95)
    assert narrow.confidence == 0.80
    assert (wide.band_high_pct - wide.band_low_pct) > (narrow.band_high_pct - narrow.band_low_pct)
    assert wide.fsu == pytest.approx(narrow.fsu, rel=1e-12)  # FSU is always at CONFIDENCE_ASHRAE


def test_monthly_model_and_monthly_reporting_work():
    rng = np.random.default_rng(42)
    model = fit_baseline(monthly_frame(rng, "2022-01-01", 24), "kwh ~ production + operating_days")
    reporting = with_saving(monthly_frame(rng, "2024-01-01", 12), 20.0)
    result = verify_savings(model, reporting)
    assert model.monthly
    assert (result.n, result.m) == (24, 12)
    assert result.savings_pct == pytest.approx(20.0, abs=1.0)  # CV(RMSE) < 1%, m = 12 bills
    assert result.verdict == "verified"


# ---------- verify_savings: the gates ----------


def test_calendar_only_noisy_model_cannot_verify_a_true_20_percent_saving():
    """kwh depends on production (sd ~170 kWh) but the model ignores it: CV(RMSE) > 0.25."""
    rng = np.random.default_rng(42)
    baseline = daily_frame(rng, "2024-01-01", 365, production_t=rng.uniform(0, 1500, 365))
    reporting = daily_frame(rng, "2025-01-01", 180, production_t=rng.uniform(0, 1500, 180))
    baseline["production"] = np.nan  # calendar-only spec (validation_plan §3, UCI-A)
    reporting["production"] = np.nan
    model = fit_baseline(baseline, CALENDAR_FORMULA)
    result = verify_savings(model, with_saving(reporting, 20.0))
    assert model.cvrmse > 0.25
    # Without the gate this band would verify, so the test is about the gate:
    assert result.band_low_pct >= 100 * ADEETIE_THRESHOLD
    assert not result.ashrae_ok
    assert result.verdict == "likely_not_proven"
    assert any("CV(RMSE)" in reason for reason in result.gate_failures)
    assert result.extrapolated_share == 0.0  # no production term, nothing to extrapolate


def test_cvrmse_gate_failure_message_names_value_and_limit():
    rng = np.random.default_rng(42)
    baseline = daily_frame(rng, n=365, production_t=rng.uniform(0, 1500, 365))
    baseline["production"] = np.nan
    model = fit_baseline(baseline, CALENDAR_FORMULA)
    reporting = baseline.iloc[:100].assign(
        period_start=pd.date_range("2025-01-01", periods=100),
        period_end=pd.date_range("2025-01-01", periods=100),
    )
    result = verify_savings(model, reporting)
    assert f"CV(RMSE) {model.cvrmse:.2f} > 0.25" in result.gate_failures


def test_fsu_gate_fails_on_a_short_noisy_monthly_reporting_period():
    """CV(RMSE) ~ 0.16 passes, but with 3 reporting bills the band is too wide to ever verify a
    10% saving: FSU at 10% saving ~ 1.2 > 0.50."""
    rng = np.random.default_rng(42)
    model = fit_baseline(
        monthly_frame(rng, "2022-01-01", 24, noise_kwh=450.0), "kwh ~ production + operating_days"
    )
    reporting = with_saving(monthly_frame(rng, "2024-01-01", 3, noise_kwh=450.0), 25.0)
    result = verify_savings(model, reporting)
    assert model.cvrmse <= 0.25
    assert result.fsu > 0.5
    assert not result.ashrae_ok
    assert result.verdict == "likely_not_proven"
    assert [reason.startswith("FSU at 10% saving") for reason in result.gate_failures] == [True]
    assert f"FSU at 10% saving {result.fsu:.2f} > 0.50" in result.gate_failures


def test_production_far_outside_the_baseline_range_is_not_proven():
    """Half of the reporting rows run at 2x the baseline maximum production."""
    rng = np.random.default_rng(42)
    baseline = daily_frame(rng, "2024-01-01", 365)
    model = fit_baseline(baseline, FORMULA)
    production_t = rng.uniform(100, 300, 180)
    production_t[:90] = 2 * baseline["production"].max()
    reporting = with_saving(daily_frame(rng, "2025-01-01", 180, production_t=production_t), 20.0)
    result = verify_savings(model, reporting)
    assert result.extrapolated_share == pytest.approx(0.5, abs=0.001)  # 90 rows + chance rows
    assert result.band_low_pct >= 100 * ADEETIE_THRESHOLD  # the band alone would verify
    assert result.ashrae_ok
    assert result.verdict == "likely_not_proven"
    assert "50% of reporting periods outside the baseline production range" in (
        result.gate_failures
    )


def test_extrapolated_share_at_the_limit_does_not_fail_the_gate():
    rng = np.random.default_rng(42)
    baseline = daily_frame(rng, "2024-01-01", 365)
    model = fit_baseline(baseline, FORMULA)
    production_t = rng.uniform(150, 250, 100)  # inside the baseline range
    production_t[:25] = 2 * baseline["production"].max()  # 25 of 100 = exactly the limit
    reporting = with_saving(daily_frame(rng, "2025-01-01", 100, production_t=production_t), 20.0)
    result = verify_savings(model, reporting)
    assert result.extrapolated_share == MAX_EXTRAPOLATED_SHARE
    assert result.gate_failures == ()


# ---------- verify_savings: exclusions and input checks ----------


def test_absurd_kwh_on_event_and_low_coverage_rows_changes_nothing():
    model, reporting = good_plant(np.random.default_rng(42))
    reporting = with_saving(reporting, 15.0)
    reporting.loc[10:19, "event"] = "shutdown"
    reporting.loc[40:49, "coverage_pct"] = 50.0
    absurd = reporting.copy()
    absurd.loc[10:19, "kwh"] = 1e9
    absurd.loc[40:49, "kwh"] = 1e9
    expected = verify_savings(model, reporting)
    assert (expected.m, expected.m_excluded) == (160, 20)
    assert verify_savings(model, absurd) == expected


def test_daily_model_with_monthly_reporting_raises():
    rng = np.random.default_rng(42)
    model = fit_baseline(daily_frame(rng, n=365), "kwh ~ production")
    with pytest.raises(ValueError, match="granularity"):
        verify_savings(model, monthly_frame(rng, "2025-01-01", 12))


def test_monthly_model_with_daily_reporting_raises():
    rng = np.random.default_rng(42)
    model = fit_baseline(monthly_frame(rng), "kwh ~ production + operating_days")
    reporting = daily_frame(rng, "2025-01-01", 30).assign(operating_days=1.0)
    with pytest.raises(ValueError, match="granularity"):
        verify_savings(model, reporting)


def test_reporting_excluded_share_above_limit_raises_insufficient_data():
    model, reporting = good_plant(np.random.default_rng(42), n_reporting=100)
    n_excluded = int(MAX_EXCLUDED_SHARE * 100) + 1  # 26 of 100 = 26% > 25%
    reporting.loc[: n_excluded - 1, "event"] = "shutdown"
    with pytest.raises(InsufficientDataError, match=r"26 of 100"):
        verify_savings(model, reporting)


def test_reporting_excluded_share_exactly_at_limit_is_allowed():
    model, reporting = good_plant(np.random.default_rng(42), n_reporting=100)
    reporting.loc[:24, "event"] = "shutdown"  # 25 of 100 = 25%, not above
    result = verify_savings(model, reporting)
    assert (result.m, result.m_excluded) == (75, 25)


def test_contract_violation_in_reporting_frame_is_reported_first():
    model, reporting = good_plant(np.random.default_rng(42))
    reporting.loc[3, "kwh"] = -1.0
    with pytest.raises(ValueError, match="kwh"):
        verify_savings(model, reporting)


def test_shuffled_reporting_rows_give_the_same_result():
    rng = np.random.default_rng(42)
    model, reporting = good_plant(rng)
    shuffled = reporting.iloc[rng.permutation(len(reporting))]
    assert verify_savings(model, shuffled) == verify_savings(model, reporting)


# ---------- SavingsResult ----------


def test_result_is_frozen_and_to_dict_round_trips_through_json():
    rng = np.random.default_rng(42)
    baseline = daily_frame(rng, n=365, production_t=rng.uniform(0, 1500, 365))
    baseline["production"] = np.nan
    model = fit_baseline(baseline, CALENDAR_FORMULA)
    result = verify_savings(model, baseline.iloc[:100])  # noisy model: has a gate failure
    assert isinstance(result, SavingsResult)
    assert result.gate_failures  # non-empty, so the list round trip below means something
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = "verified"
    d = result.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["gate_failures"] == list(result.gate_failures)
    assert d["verdict"] == "likely_not_proven"
    assert {"extrapolated_share", "m_excluded", "ashrae_ok", "fsu"} <= d.keys()
