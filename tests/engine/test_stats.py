"""Tests for engine/stats.py: ASHRAE Guideline 14 statistics (formulas: mv-engine reference)."""

import inspect

import numpy as np
import pytest
from scipy import stats as scipy_stats

from saveproof.engine.stats import (
    cvrmse,
    lag1_autocorr,
    n_effective,
    nmbe,
    savings_uncertainty_kwh,
)

# Hand example, n = 6 rows, p = 2 parameters (intercept + slope).
Y_KWH = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
YHAT_KWH = np.array([9.0, 7.0, 12.0, 18.0, 19.0, 19.0])
P = 2


def test_hand_example_residuals_are_what_the_docstrings_assume():
    """Residuals e = y - yhat = [1, 5, 2, -2, -1, 1]; the arithmetic below builds on them."""
    np.testing.assert_allclose(Y_KWH - YHAT_KWH, [1, 5, 2, -2, -1, 1])


def test_cvrmse_hand_example():
    """CV(RMSE) = sqrt(sum e^2 / (n - p)) / mean(y), by hand:

    sum e^2 = 1 + 25 + 4 + 4 + 1 + 1 = 36
    n - p   = 6 - 2 = 4
    RMSE    = sqrt(36 / 4) = 3
    mean(y) = (10 + 12 + 14 + 16 + 18 + 20) / 6 = 90 / 6 = 15
    CV(RMSE) = 3 / 15 = 0.2
    """
    assert cvrmse(Y_KWH, YHAT_KWH, P) == pytest.approx(0.2, abs=1e-12)


def test_nmbe_hand_example():
    """NMBE = sum e / ((n - p) * mean(y)), by hand:

    sum e = 1 + 5 + 2 - 2 - 1 + 1 = 6
    (n - p) * mean(y) = 4 * 15 = 60
    NMBE = 6 / 60 = 0.1   (positive: the model under-predicts on average)
    """
    assert nmbe(Y_KWH, YHAT_KWH, P) == pytest.approx(0.1, abs=1e-12)


def test_lag1_autocorr_hand_example():
    """rho = Pearson correlation of e[1:] with e[:-1] (5 pairs), by hand:

    e[1:]  (current) = [5, 2, -2, -1, 1]   mean = 5 / 5 = 1
    e[:-1] (lagged)  = [1, 5, 2, -2, -1]   mean = 5 / 5 = 1
    deviations current = [4, 1, -3, -2, 0]
    deviations lagged  = [0, 4, 1, -3, -2]
    sum of products = 0 + 4 - 3 + 6 + 0 = 7
    sum sq current  = 16 + 1 + 9 + 4 + 0 = 30
    sum sq lagged   = 0 + 16 + 1 + 9 + 4 = 30
    rho = 7 / sqrt(30 * 30) = 7 / 30 = 0.23333...
    """
    assert lag1_autocorr(Y_KWH - YHAT_KWH) == pytest.approx(7 / 30, abs=1e-12)


def test_n_effective_hand_example():
    """n' = n (1 - rho) / (1 + rho), by hand with n = 6 and rho = 7/30:

    1 - rho = 23/30,  1 + rho = 37/30
    n' = 6 * (23/30) / (37/30) = 6 * 23 / 37 = 138 / 37 = 3.72973...
    """
    assert n_effective(6, 7 / 30) == pytest.approx(138 / 37, abs=1e-12)


def test_n_effective_zero_rho_gives_n():
    assert n_effective(365, 0.0) == pytest.approx(365.0)


@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_n_effective_positive_rho_gives_fewer_than_n(rho):
    assert n_effective(100, rho) < 100


@pytest.mark.parametrize("rho", [-1.0, 1.0, -1.5, 1.5])
def test_n_effective_rho_outside_open_unit_interval_raises(rho):
    with pytest.raises(ValueError, match="rho"):
        n_effective(100, rho)


# ---- savings_uncertainty_kwh -------------------------------------------------

# One scenario used for all uncertainty tests: 365 baseline days, 2-parameter model,
# 90 reporting days, adjusted baseline 90 000 kWh (= 1 000 kWh/day), CV(RMSE) 8%.
U = dict(cv=0.08, n=365, p=2, rho=0.0, m=90, adjusted_baseline_kwh=90_000.0, confidence=0.90)


def _t_two_sided(confidence, dof):
    return scipy_stats.t.ppf(1 - (1 - confidence) / 2, dof)


def test_uncertainty_takes_no_savings_argument_and_is_deterministic():
    """The band half-width must not depend on how big the saving is: no savings parameter exists."""
    params = inspect.signature(savings_uncertainty_kwh).parameters
    assert not [name for name in params if "saving" in name]
    first = savings_uncertainty_kwh(**U, monthly=False)
    second = savings_uncertainty_kwh(**U, monthly=False)
    assert first == second


def test_uncertainty_monthly_uses_1_26_factor():
    """monthly=True: dE = 1.26 * t * CV * sqrt((n/n') * (1 + 2/n) / m) * E_adjusted_baseline.

    With rho = 0, n' = n so the n/n' term is 1. t is two-sided at (n - p) degrees of freedom.
    """
    t = _t_two_sided(0.90, U["n"] - U["p"])
    expected = 1.26 * t * U["cv"] * np.sqrt((1 + 2 / U["n"]) / U["m"]) * U["adjusted_baseline_kwh"]
    assert savings_uncertainty_kwh(**U, monthly=True) == pytest.approx(expected, rel=1e-12)


def test_uncertainty_monthly_includes_autocorrelation_inflation():
    """With rho = 0.5: n' = n/3, so n/n' = 3 and the monthly half-width grows by sqrt(3)."""
    rho_zero = savings_uncertainty_kwh(**U, monthly=True)
    rho_half = savings_uncertainty_kwh(**{**U, "rho": 0.5}, monthly=True)
    assert rho_half == pytest.approx(rho_zero * np.sqrt(3), rel=1e-12)


def test_uncertainty_daily_does_not_use_1_26():
    """monthly=False: nmecr daily form (calculate_savings_and_uncertainty.R, kW-Labs, MIT).

    alpha = -0.00024 x^2 + 0.03535 x + 1.00286 with x = m / 30.5 replaces 1.26. With rho = 0,
    n' = n, so the rest equals the monthly expression.
    """
    x = U["m"] / 30.5
    alpha = -0.00024 * x**2 + 0.03535 * x + 1.00286
    t = _t_two_sided(0.90, U["n"] - U["p"])
    expected = alpha * t * U["cv"] * np.sqrt((1 + 2 / U["n"]) / U["m"]) * U["adjusted_baseline_kwh"]
    assert savings_uncertainty_kwh(**U, monthly=False) == pytest.approx(expected, rel=1e-12)
    assert savings_uncertainty_kwh(**U, monthly=False) != pytest.approx(
        savings_uncertainty_kwh(**U, monthly=True), rel=1e-6
    )


def test_uncertainty_daily_with_autocorrelation_matches_nmecr_sse_form():
    """Cross-check the CV-based daily form against nmecr's SSE-based algebra.

    nmecr: dE = t' * alpha * sqrt(MSE' * (1 + 2/n') * m), MSE' = SSE / (n' - p),
    t' at (n' - p) degrees of freedom. Here SSE is rebuilt from cv, so the two algebras differ.
    """
    rho = 0.6
    n, p, m, cv, e_adj = 365, 2, 90, 0.08, 90_000.0
    mean_kwh = e_adj / m
    sse = (cv * mean_kwh) ** 2 * (n - p)
    n_dash = n * (1 - rho) / (1 + rho)
    mse_dash = sse / (n_dash - p)
    x = m / 30.5
    alpha = -0.00024 * x**2 + 0.03535 * x + 1.00286
    expected = _t_two_sided(0.90, n_dash - p) * alpha * np.sqrt(mse_dash * (1 + 2 / n_dash) * m)
    got = savings_uncertainty_kwh(cv, n, p, rho, m, e_adj, 0.90, monthly=False)
    assert got == pytest.approx(expected, rel=1e-12)


def test_uncertainty_scales_linearly_with_adjusted_baseline():
    base = savings_uncertainty_kwh(**U, monthly=False)
    doubled = savings_uncertainty_kwh(**{**U, "adjusted_baseline_kwh": 180_000.0}, monthly=False)
    assert doubled == pytest.approx(2 * base, rel=1e-12)


def test_uncertainty_higher_confidence_widens_band():
    assert savings_uncertainty_kwh(**{**U, "confidence": 0.95}, monthly=False) > (
        savings_uncertainty_kwh(**{**U, "confidence": 0.68}, monthly=False)
    )


def test_uncertainty_daily_needs_positive_dof_after_autocorrelation():
    """n' = 20 * 0.01 / 1.99 ~ 0.1 <= p, so the daily form has no degrees of freedom."""
    with pytest.raises(ValueError, match="rho"):
        savings_uncertainty_kwh(**{**U, "n": 20, "rho": 0.99}, monthly=False)


@pytest.mark.parametrize("monthly", [True, False])
def test_uncertainty_negative_rho_is_clamped_to_zero(monthly):
    """rho = -0.3 would make n' > n and narrow the band; it must give the rho = 0 band."""
    clamped = savings_uncertainty_kwh(**{**U, "rho": -0.3}, monthly=monthly)
    assert clamped == pytest.approx(savings_uncertainty_kwh(**U, monthly=monthly), rel=1e-12)


@pytest.mark.parametrize("monthly", [True, False])
def test_uncertainty_has_no_jump_at_rho_one_half(monthly):
    """nmecr switches to n' at rho >= 0.5 (band jumps ~1.8x); ours is continuous across 0.5."""
    below = savings_uncertainty_kwh(**{**U, "rho": 0.499}, monthly=monthly)
    above = savings_uncertainty_kwh(**{**U, "rho": 0.501}, monthly=monthly)
    assert abs(above - below) / below < 0.01


@pytest.mark.parametrize("monthly", [True, False])
def test_uncertainty_increases_with_rho(monthly):
    bands = [
        savings_uncertainty_kwh(**{**U, "rho": r}, monthly=monthly) for r in (0, 0.2, 0.4, 0.6)
    ]
    assert bands == sorted(bands)
    assert len(set(bands)) == len(bands)


@pytest.mark.parametrize("monthly", [True, False])
def test_uncertainty_rho_at_or_below_minus_one_still_raises(monthly):
    """Clamping must not hide an invalid rho."""
    with pytest.raises(ValueError, match=r"\brho\b"):
        savings_uncertainty_kwh(**{**U, "rho": -1.0}, monthly=monthly)


def test_uncertainty_daily_accepts_a_full_year_of_reporting_days():
    """m = 366 gives x = 366 / 30.5 = 12.0 months, the upper limit of the daily factor."""
    band_kwh = savings_uncertainty_kwh(**{**U, "m": 366}, monthly=False)
    assert np.isfinite(band_kwh)
    assert band_kwh > 0


def test_uncertainty_daily_rejects_more_than_twelve_months_of_reporting_days():
    with pytest.raises(ValueError, match=r"\bm\b.*1-12 month"):
        savings_uncertainty_kwh(**{**U, "m": 400}, monthly=False)


@pytest.mark.parametrize(
    ("override", "arg"),
    [
        ({"n": 2}, "n"),  # n <= p
        ({"cv": -0.1}, "cv"),
        ({"m": 0}, "m"),
        ({"adjusted_baseline_kwh": -1.0}, "adjusted_baseline_kwh"),
        ({"confidence": 1.0}, "confidence"),
        ({"confidence": 0.0}, "confidence"),
        ({"rho": 1.0}, "rho"),
    ],
)
def test_uncertainty_bad_input_raises_naming_the_argument(override, arg):
    # whole-word match, so "m" cannot pass by matching a letter inside another name
    with pytest.raises(ValueError, match=rf"\b{arg}\b"):
        savings_uncertainty_kwh(**{**U, **override}, monthly=False)


# ---- input validation for the array statistics ---------------------------------


@pytest.mark.parametrize("func", [cvrmse, nmbe])
def test_different_length_raises_naming_yhat(func):
    with pytest.raises(ValueError, match="yhat"):
        func(Y_KWH, YHAT_KWH[:-1], P)


@pytest.mark.parametrize("func", [cvrmse, nmbe])
@pytest.mark.parametrize("p", [6, 7])
def test_n_not_greater_than_p_raises_naming_p(func, p):
    with pytest.raises(ValueError, match=r"\bp\b"):
        func(Y_KWH, YHAT_KWH, p)


@pytest.mark.parametrize("func", [cvrmse, nmbe])
def test_non_finite_input_raises_naming_the_array_and_row(func):
    bad = Y_KWH.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match=r"y.*row 3"):
        func(bad, YHAT_KWH, P)


@pytest.mark.parametrize("func", [cvrmse, nmbe])
def test_zero_mean_y_raises_naming_y(func):
    with pytest.raises(ValueError, match=r"\by\b"):
        func(np.zeros(6), YHAT_KWH, P)


def test_lag1_autocorr_needs_three_points_and_variance():
    with pytest.raises(ValueError, match="resid"):
        lag1_autocorr(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="resid"):
        lag1_autocorr(np.array([3.0, 3.0, 3.0, 3.0]))


@pytest.mark.parametrize("n", [0, -5])
def test_n_effective_non_positive_n_raises(n):
    with pytest.raises(ValueError, match=r"\bn\b"):
        n_effective(n, 0.0)
