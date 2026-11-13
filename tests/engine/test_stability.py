"""Tests for engine/stability.py: baseline stability flag, docs/validation_addendum_v5b.md §10.

All frames are synthetic and built here with rng = np.random.default_rng(42).
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from saveproof.config import STABILITY_DRIFT_SHARE_MIN, STABILITY_P_MAX
from saveproof.engine.baseline import fit_baseline
from saveproof.engine.stability import (
    StabilityResult,
    baseline_stability,
    is_unstable,
    newey_west_lag,
)

FORMULA = "kwh ~ production"


def daily_frame(rng, n=365, trend_kwh_per_day=0.0, noise_kwh=5.0, dates=None):
    """kwh = 180 + 0.4*production + trend*(days since the first date) + noise; mean ~ 260 kWh."""
    dates = pd.DatetimeIndex(dates) if dates is not None else pd.date_range("2025-01-01", periods=n)
    n = len(dates)
    production_t = rng.uniform(100, 300, n)
    days = (dates - dates[0]).days.to_numpy()
    return pd.DataFrame(
        {
            "period_start": dates,
            "period_end": dates,
            "kwh": 180
            + 0.4 * production_t
            + trend_kwh_per_day * days
            + rng.normal(0, noise_kwh, n),
            "production": production_t,
            "production_unit": "t",
            "operating_days": 1.0,
            "source": "simulated",
            "dataset": "test",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


def stability_of(frame):
    return baseline_stability(fit_baseline(frame, FORMULA))


def test_thresholds_are_the_addendum_values():
    assert STABILITY_P_MAX == 0.05  # addendum §10, significance
    assert STABILITY_DRIFT_SHARE_MIN == 0.05  # addendum §10, materiality


@pytest.mark.parametrize(
    ("n", "lag"),
    [
        (14, 2),  # 4 * 0.14 ** (2/9) = 4 * 0.646 = 2.58
        (20, 2),  # 4 * 0.20 ** (2/9) = 4 * 0.699 = 2.80
        (100, 4),  # 4 * 1 = 4 exactly
        (243, 4),  # 4 * 2.43 ** (2/9) = 4 * 1.218 = 4.87
        (365, 5),  # 4 * 3.65 ** (2/9) = 4 * 1.333 = 5.33
    ],
)
def test_newey_west_lag_is_floor_of_4_n_over_100_to_2_ninths(n, lag):
    assert newey_west_lag(n) == lag


@pytest.mark.parametrize(
    ("p_value", "drift_share", "flagged"),
    [
        (0.01, 0.10, True),
        (0.0499, 0.05, True),  # drift exactly at the materiality threshold counts
        (0.05, 0.10, False),  # p exactly 0.05 is not significant (p < 0.05)
        (0.01, 0.0499, False),  # significant but immaterial
        (0.50, 0.50, False),  # material but not significant
    ],
)
def test_flag_needs_both_significance_and_materiality(p_value, drift_share, flagged):
    assert is_unstable(p_value, drift_share) is flagged


def test_no_trend_is_not_flagged():
    """Pure noise: |slope| ~ 5 / sqrt(365^3 / 12) ~ 0.0025 kWh/day, so D ~ 0.004 << 0.05."""
    result = stability_of(daily_frame(np.random.default_rng(42)))
    assert result.drift_share < STABILITY_DRIFT_SHARE_MIN
    assert result.flagged is False


def test_large_planted_trend_is_flagged_and_its_slope_recovered():
    """0.1 kWh/day over 364 days = 36.4 kWh of drift on a ~270 kWh mean: D ~ 0.135.
    Slope standard error ~ 0.0025 kWh/day, so abs=0.01 is ~4 standard errors."""
    frame = daily_frame(np.random.default_rng(42), trend_kwh_per_day=0.1)
    result = stability_of(frame)
    assert result.slope_kwh_per_day == pytest.approx(0.1, abs=0.01)
    assert result.p_value < 1e-6
    assert result.drift_share == pytest.approx(
        abs(result.slope_kwh_per_day) * 364 / frame["kwh"].mean(), rel=1e-12
    )
    assert result.drift_share > 0.1
    assert result.flagged is True


def test_significant_but_tiny_trend_is_not_flagged():
    """0.02 kWh/day: t-statistic ~ 0.02 / 0.0025 = 8 (p << 0.05) but D ~ 7.3 / 264 ~ 0.028."""
    result = stability_of(daily_frame(np.random.default_rng(42), trend_kwh_per_day=0.02))
    assert result.p_value < STABILITY_P_MAX
    assert result.drift_share < STABILITY_DRIFT_SHARE_MIN
    assert result.flagged is False


def test_material_but_insignificant_trend_is_not_flagged():
    """20 days, 2 kWh/day under 40 kWh noise: D ~ 38 / 280 ~ 0.14, slope SE ~ 40 / 26 ~ 1.5."""
    result = stability_of(
        daily_frame(np.random.default_rng(42), n=20, trend_kwh_per_day=2.0, noise_kwh=40.0)
    )
    assert result.drift_share >= STABILITY_DRIFT_SHARE_MIN
    assert result.p_value >= STABILITY_P_MAX
    assert result.flagged is False


def test_falling_trend_is_flagged_like_a_rising_one():
    result = stability_of(daily_frame(np.random.default_rng(42), trend_kwh_per_day=-0.1))
    assert result.slope_kwh_per_day == pytest.approx(-0.1, abs=0.01)
    assert result.drift_share > 0.1
    assert result.flagged is True


def test_t_is_calendar_days_not_row_number():
    """100 dates with a 100-day gap after the 50th: the slope is per day, not per row (per row it
    would be ~ 0.1 * 200 / 100 = 0.2)."""
    dates = pd.date_range("2025-01-01", periods=50).append(pd.date_range("2025-06-01", periods=50))
    result = stability_of(
        daily_frame(np.random.default_rng(42), trend_kwh_per_day=0.1, dates=dates)
    )
    assert result.slope_kwh_per_day == pytest.approx(0.1, abs=0.01)


def test_shuffled_input_rows_give_the_same_result():
    rng = np.random.default_rng(42)
    frame = daily_frame(rng, trend_kwh_per_day=0.05)
    shuffled = frame.iloc[rng.permutation(len(frame))]
    assert stability_of(shuffled) == stability_of(frame)


def test_p_value_matches_a_hand_computed_newey_west_test():
    """Newey-West (1987): Bartlett weights w_j = 1 - j / (L + 1), no small-sample correction,
    normal reference distribution, two-sided. Computed here directly from the residuals.
    n = 20 -> L = 2."""
    model = fit_baseline(
        daily_frame(np.random.default_rng(42), n=20, trend_kwh_per_day=0.5), FORMULA
    )
    e = model.result.resid.to_numpy()
    t = np.arange(20.0)  # contiguous dates, so days since start = 0..19
    x = np.column_stack([np.ones(20), t])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ e
    u = e - x @ beta
    lag = 2
    meat = (x * u[:, None]).T @ (x * u[:, None])
    for j in range(1, lag + 1):
        weight = 1 - j / (lag + 1)
        cross = (x[j:] * u[j:, None]).T @ (x[:-j] * u[:-j, None])
        meat += weight * (cross + cross.T)
    se_slope = np.sqrt((xtx_inv @ meat @ xtx_inv)[1, 1])
    p_value = 2 * scipy_stats.norm.sf(abs(beta[1]) / se_slope)

    result = baseline_stability(model)
    assert result.slope_kwh_per_day == pytest.approx(beta[1], rel=1e-9)
    assert result.p_value == pytest.approx(p_value, rel=1e-9)
    assert result.drift_share == pytest.approx(
        abs(beta[1]) * 19 / model.result.model.data.frame["kwh"].mean(), rel=1e-12
    )


def test_monthly_baseline_raises():
    """D = |slope| x (n - 1) assumes one period per day (addendum §10)."""
    starts = pd.date_range("2022-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(42)
    production_t = rng.uniform(2000, 6000, 24)
    monthly = daily_frame(rng, n=24).assign(
        period_start=starts,
        period_end=starts + pd.offsets.MonthEnd(0),
        production=production_t,
        kwh=500 + 0.4 * production_t + rng.normal(0, 20, 24),
        operating_days=20.0,
        source="utility_bill",
    )
    with pytest.raises(ValueError, match="daily"):
        baseline_stability(fit_baseline(monthly, FORMULA))


def test_result_is_frozen_and_json_serialisable():
    result = stability_of(daily_frame(np.random.default_rng(42), n=30))
    assert isinstance(result, StabilityResult)
    d = result.to_dict()
    assert set(d) == {"slope_kwh_per_day", "p_value", "drift_share", "flagged"}
    assert json.loads(json.dumps(d)) == d
    assert isinstance(d["flagged"], bool)
