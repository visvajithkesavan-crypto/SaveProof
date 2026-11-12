"""Tests for engine/baseline.py: OLS baseline on usable rows, with ASHRAE G14 statistics.

All frames are synthetic and built here with rng = np.random.default_rng(42).
"""

import json

import numpy as np
import pandas as pd
import pytest

from saveproof.config import MAX_EXCLUDED_SHARE, MIN_COVERAGE_PCT
from saveproof.engine import stats
from saveproof.engine.baseline import (
    InsufficientDataError,
    add_calendar_terms,
    fit_baseline,
    predict,
    usable_rows,
)

PROD_FORMULA = "kwh ~ production"
DAY_TYPE_FORMULA = "kwh ~ production + C(day_type)"


def daily_frame(rng, n=365, intercept_kwh=180.0, slope_kwh_per_t=0.4, weekend_kwh=0.0):
    """Daily contract table starting Wed 2025-01-01: kwh = a + b*production + weekend + noise."""
    start = pd.date_range("2025-01-01", periods=n, freq="D")
    production_t = rng.uniform(100, 300, n)
    weekend = (start.dayofweek >= 5).astype(float)
    noise_kwh = rng.normal(0, 5.0, n)
    return pd.DataFrame(
        {
            "period_start": start,
            "period_end": start,
            "kwh": intercept_kwh
            + slope_kwh_per_t * production_t
            + weekend_kwh * weekend
            + noise_kwh,
            "production": production_t,
            "production_unit": "t",
            "operating_days": 1.0,
            "source": "simulated",
            "dataset": "test",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


def monthly_frame(rng, n=24):
    """Monthly bills: kwh = 500 + 0.4*production + 30*operating_days + noise."""
    start = pd.date_range("2024-01-01", periods=n, freq="MS")
    end = start + pd.offsets.MonthEnd(0)
    operating_days = rng.integers(20, 27, n).astype(float)
    production_t = rng.uniform(2000, 6000, n)
    noise_kwh = rng.normal(0, 20.0, n)
    return pd.DataFrame(
        {
            "period_start": start,
            "period_end": end,
            "kwh": 500 + 0.4 * production_t + 30 * operating_days + noise_kwh,
            "production": production_t,
            "production_unit": "t",
            "operating_days": operating_days,
            "source": "simulated",
            "dataset": "test",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


def bills_frame(rng, n=12):
    """Irregular billing cycles of 28-35 days, back to back from 2024-01-01.
    kwh = 500 + 0.4*production + 30*operating_days + 10*period_days + noise."""
    period_days = rng.integers(28, 36, n)
    start = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.cumsum(period_days) - period_days, "D")
    end = start + pd.to_timedelta(period_days - 1, "D")
    operating_days = (period_days - rng.integers(3, 9, n)).astype(float)
    production_t = rng.uniform(2000, 6000, n)
    noise_kwh = rng.normal(0, 20.0, n)
    return pd.DataFrame(
        {
            "period_start": start,
            "period_end": end,
            "kwh": 500 + 0.4 * production_t + 30 * operating_days + 10 * period_days + noise_kwh,
            "production": production_t,
            "production_unit": "t",
            "operating_days": operating_days,
            "source": "utility_bill",
            "dataset": "test",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


# ---------- add_calendar_terms ----------


def test_add_calendar_terms_period_days_is_1_for_daily_and_counts_a_30_day_bill():
    daily = daily_frame(np.random.default_rng(42), n=3)
    assert add_calendar_terms(daily)["period_days"].tolist() == [1, 1, 1]
    bill = daily.iloc[:1].assign(
        period_start=pd.Timestamp("2025-04-01"), period_end=pd.Timestamp("2025-04-30")
    )
    assert add_calendar_terms(bill)["period_days"].tolist() == [30]  # end date inclusive


def test_add_calendar_terms_marks_saturday_and_sunday_and_does_not_mutate():
    df = daily_frame(np.random.default_rng(42), n=7)  # Wed 1 Jan .. Tue 7 Jan 2025
    out = add_calendar_terms(df)
    assert "period_days" not in df.columns
    assert "day_type" not in df.columns
    # Wed Thu Fri Sat Sun Mon Tue
    assert out["day_type"].tolist() == [
        "weekday",
        "weekday",
        "weekday",
        "weekend",
        "weekend",
        "weekday",
        "weekday",
    ]


# ---------- usable_rows ----------


def test_usable_rows_excludes_events_low_coverage_and_nan_regressors():
    df = daily_frame(np.random.default_rng(42), n=6)
    df.loc[1, "event"] = "shutdown"
    df.loc[2, "coverage_pct"] = MIN_COVERAGE_PCT - 0.1
    df.loc[3, "coverage_pct"] = MIN_COVERAGE_PCT  # exactly at the limit is usable
    df.loc[4, "production"] = np.nan
    mask = usable_rows(df, PROD_FORMULA)
    assert mask.tolist() == [True, False, False, True, False, True]
    assert mask.index.equals(df.index)


def test_usable_rows_ignores_nan_in_columns_the_formula_does_not_use():
    df = daily_frame(np.random.default_rng(42), n=4)
    df.loc[1, "production"] = np.nan
    assert usable_rows(df, "kwh ~ C(day_type)").all()  # calendar-only spec, production is NaN


# ---------- fit_baseline: accuracy ----------


def test_known_answer_recovers_intercept_and_slope():
    """kwh = 180 + 0.4*production + N(0, 5). OLS is unbiased, so each estimate lies within
    3 standard errors of the truth with probability ~99.7%; the seed is fixed, so this is
    deterministic."""
    df = daily_frame(np.random.default_rng(42), n=365)
    model = fit_baseline(df, PROD_FORMULA)
    bse = model.result.bse
    assert abs(model.params["Intercept"] - 180.0) < 3 * bse["Intercept"]
    assert abs(model.params["production"] - 0.4) < 3 * bse["production"]
    assert model.n == 365
    assert model.p == 2
    assert model.monthly is False
    assert model.n_excluded == 0
    assert model.production_min == pytest.approx(df["production"].min())
    assert model.production_max == pytest.approx(df["production"].max())


def test_weekend_effect_recovered_with_day_type_term():
    df = daily_frame(np.random.default_rng(42), n=365, weekend_kwh=-60.0)
    model = fit_baseline(df, DAY_TYPE_FORMULA)
    name = "C(day_type)[T.weekend]"
    # Same 3-standard-error argument as the known-answer test.
    assert abs(model.params[name] - (-60.0)) < 3 * model.result.bse[name]


def test_production_range_is_nan_when_formula_has_no_production():
    df = daily_frame(np.random.default_rng(42), n=100)
    model = fit_baseline(df, "kwh ~ C(day_type)")
    assert np.isnan(model.production_min)
    assert np.isnan(model.production_max)


def test_monthly_fit_with_period_days_works_on_28_to_35_day_bills():
    df = bills_frame(np.random.default_rng(42), n=12)
    assert df["period_end"].sub(df["period_start"]).dt.days.add(1).nunique() > 1  # irregular
    model = fit_baseline(df, "kwh ~ production + operating_days + period_days")
    assert model.monthly is True
    assert model.p == 4
    assert model.n == 12
    assert "period_days" in model.params


def test_monthly_formula_with_day_type_raises():
    df = monthly_frame(np.random.default_rng(42), n=12)
    with pytest.raises(ValueError, match="day_type") as excinfo:
        fit_baseline(df, DAY_TYPE_FORMULA)
    assert "multi-day" in str(excinfo.value)
    assert "period_days" in str(excinfo.value)


def test_monthly_bills_set_monthly_flag_and_fit_default_formula():
    model = fit_baseline(monthly_frame(np.random.default_rng(42)))
    assert model.monthly is True
    assert model.formula == "kwh ~ production + operating_days"
    assert model.p == 3
    assert model.n == 24


# ---------- fit_baseline: exclusions ----------


def test_excluded_rows_do_not_influence_the_fit():
    rng = np.random.default_rng(42)
    clean = daily_frame(rng, n=365)
    clean.loc[10:19, "event"] = "shutdown"
    clean.loc[40:49, "coverage_pct"] = 50.0
    absurd = clean.copy()
    absurd.loc[10:19, "kwh"] = 1e9
    absurd.loc[40:49, "kwh"] = 1e9
    reference = fit_baseline(clean, PROD_FORMULA)
    model = fit_baseline(absurd, PROD_FORMULA)
    assert model.n_excluded == 20
    assert model.n == 345
    for name, value in reference.params.items():
        assert model.params[name] == pytest.approx(value, rel=1e-9)
    assert model.cvrmse == pytest.approx(reference.cvrmse, rel=1e-9)


def test_excluded_share_above_limit_raises_insufficient_data():
    df = daily_frame(np.random.default_rng(42), n=100)
    n_excluded = int(MAX_EXCLUDED_SHARE * 100) + 1  # 26 of 100 = 26% > 25%
    df.loc[: n_excluded - 1, "event"] = "shutdown"
    with pytest.raises(InsufficientDataError, match=r"26 of 100"):
        fit_baseline(df, PROD_FORMULA)


def test_excluded_share_exactly_at_limit_is_allowed():
    df = daily_frame(np.random.default_rng(42), n=100)
    df.loc[:24, "event"] = "shutdown"  # 25 of 100 = 25%, not above
    assert fit_baseline(df, PROD_FORMULA).n == 75


def test_too_few_degrees_of_freedom_raises_insufficient_data():
    df = daily_frame(np.random.default_rng(42), n=6)  # n - p = 6 - 2 = 4 < 5
    with pytest.raises(InsufficientDataError, match=r"n - p"):
        fit_baseline(df, PROD_FORMULA)


def test_insufficient_data_is_a_value_error():
    assert issubclass(InsufficientDataError, ValueError)


# ---------- fit_baseline: input checks ----------


def test_daily_data_with_default_formula_names_operating_days():
    df = daily_frame(np.random.default_rng(42), n=100)
    with pytest.raises(ValueError, match="operating_days") as excinfo:
        fit_baseline(df)
    assert "C(day_type)" in str(excinfo.value)


def test_collinear_terms_raise_rather_than_silently_pseudo_inverting():
    df = daily_frame(np.random.default_rng(42), n=100)
    with pytest.raises(ValueError, match="collinear"):
        fit_baseline(df, "kwh ~ production + I(production * 2)")


def test_mixed_daily_and_monthly_periods_raise():
    df = monthly_frame(np.random.default_rng(42))
    df.loc[0, "period_end"] = df.loc[0, "period_start"]  # one 1-day row among monthly rows
    df.loc[0, "operating_days"] = 1.0
    with pytest.raises(ValueError, match="mixed"):
        fit_baseline(df)


def test_contract_violation_is_reported_before_fitting():
    df = daily_frame(np.random.default_rng(42), n=100)
    df.loc[3, "kwh"] = -1.0
    with pytest.raises(ValueError, match="kwh"):
        fit_baseline(df, PROD_FORMULA)


# ---------- invariants ----------


def test_shuffled_input_gives_same_params():
    rng = np.random.default_rng(42)
    df = daily_frame(rng, n=200)
    shuffled = df.iloc[rng.permutation(len(df))]
    a = fit_baseline(df, DAY_TYPE_FORMULA)
    b = fit_baseline(shuffled, DAY_TYPE_FORMULA)
    for name, value in a.params.items():
        assert b.params[name] == pytest.approx(value, rel=1e-9)
    assert b.rho == pytest.approx(a.rho, rel=1e-9)  # residual order is period order, not row order


def test_scaling_kwh_scales_params_and_leaves_cvrmse_unchanged():
    k = 3.0
    df = daily_frame(np.random.default_rng(42), n=200)
    scaled = df.assign(kwh=df["kwh"] * k)
    a = fit_baseline(df, DAY_TYPE_FORMULA)
    b = fit_baseline(scaled, DAY_TYPE_FORMULA)
    for name, value in a.params.items():
        assert b.params[name] == pytest.approx(k * value, rel=1e-9)
    assert b.cvrmse == pytest.approx(a.cvrmse, rel=1e-9)
    assert b.r2 == pytest.approx(a.r2, rel=1e-9)


# ---------- statistics come from stats.py ----------


def test_model_statistics_equal_stats_module_on_usable_rows():
    rng = np.random.default_rng(42)
    df = daily_frame(rng, n=200)
    df.loc[5:14, "event"] = "meter_fault"
    model = fit_baseline(df, DAY_TYPE_FORMULA)
    rows = df[usable_rows(df, DAY_TYPE_FORMULA)]
    y_kwh = rows["kwh"].to_numpy()
    yhat_kwh = predict(model, rows).to_numpy()
    assert model.cvrmse == pytest.approx(stats.cvrmse(y_kwh, yhat_kwh, model.p), rel=1e-12)
    assert model.nmbe == pytest.approx(stats.nmbe(y_kwh, yhat_kwh, model.p), abs=1e-12)
    rho = stats.lag1_autocorr(y_kwh - yhat_kwh)
    assert model.rho == pytest.approx(rho, rel=1e-9)
    assert model.n_eff == pytest.approx(stats.n_effective(model.n, max(rho, 0.0)), rel=1e-9)


def test_negative_rho_does_not_increase_n_eff():
    """Alternating residuals give rho < 0; n' is computed from max(rho, 0), so n' == n."""
    df = daily_frame(np.random.default_rng(42), n=100)
    df["kwh"] = 180 + 0.4 * df["production"] + np.where(np.arange(100) % 2 == 0, 5.0, -5.0)
    model = fit_baseline(df, PROD_FORMULA)
    assert model.rho < 0
    assert model.n_eff == pytest.approx(model.n)


# ---------- predict ----------


def test_predict_on_baseline_rows_equals_fitted_values():
    df = daily_frame(np.random.default_rng(42), n=120)
    model = fit_baseline(df, DAY_TYPE_FORMULA)
    pred = predict(model, df)
    assert pred.name == "kwh_pred"
    np.testing.assert_allclose(pred.to_numpy(), model.result.fittedvalues.to_numpy(), rtol=1e-10)


def test_predict_is_aligned_to_the_input_index():
    df = daily_frame(np.random.default_rng(42), n=50)
    model = fit_baseline(df, PROD_FORMULA)
    odd = df.iloc[::-1].set_axis(np.arange(1000, 1050))
    pred = predict(model, odd)
    assert pred.index.equals(odd.index)
    expected = model.params["Intercept"] + model.params["production"] * odd["production"]
    np.testing.assert_allclose(pred.to_numpy(), expected.to_numpy(), rtol=1e-10)


def test_predict_unseen_day_type_level_raises_naming_it():
    df = daily_frame(np.random.default_rng(42), n=120)
    weekdays_only = df[pd.to_datetime(df["period_start"]).dt.dayofweek < 5]
    model = fit_baseline(weekdays_only, DAY_TYPE_FORMULA)
    with pytest.raises(ValueError, match="weekend"):
        predict(model, df)


def test_predict_nan_regressor_raises_naming_the_column():
    df = daily_frame(np.random.default_rng(42), n=60)
    model = fit_baseline(df, PROD_FORMULA)
    reporting = df.copy()
    reporting.loc[7, "production"] = np.nan
    with pytest.raises(ValueError, match="production"):
        predict(model, reporting)


# ---------- to_dict ----------


def test_to_dict_round_trips_through_json_and_excludes_the_result():
    df = daily_frame(np.random.default_rng(42), n=100)
    model = fit_baseline(df, DAY_TYPE_FORMULA)
    d = model.to_dict()
    assert "result" not in d
    assert json.loads(json.dumps(d)) == d
    assert d["formula"] == DAY_TYPE_FORMULA
    assert d["n"] == 100


def test_model_records_the_baseline_production_unit():
    """The proxy gate (addendum §9) reads it from the model."""
    df = daily_frame(np.random.default_rng(42), n=100).assign(production_unit="proxy:pv_kwh")
    assert fit_baseline(df, DAY_TYPE_FORMULA).production_unit == "proxy:pv_kwh"
    assert fit_baseline(df, DAY_TYPE_FORMULA).to_dict()["production_unit"] == "proxy:pv_kwh"
