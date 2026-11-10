"""Tests for engine/comparators.py: naive bill comparison and BEE SEC-ratio, the two methods
SaveProof is compared against (docs/validation_plan.md §2: no band, verdict from the point
estimate).

All frames are synthetic and built here with rng = np.random.default_rng(42).
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from saveproof.engine.baseline import fit_baseline
from saveproof.engine.comparators import ComparatorResult, bee_sec_savings, naive_savings
from saveproof.engine.savings import verify_savings

FORMULA = "kwh ~ production"
BASE_LOAD_KWH = 2000.0  # fixed daily load: lighting, idling compressors, furnace holding
VARIABLE_SEC_KWH_PER_T = 15.0


def plant_frame(rng, start, production_t, noise_kwh=20.0):
    """Daily contract table: kwh = 2000 + 15*production + noise (noise ~0.6% of ~3500 kWh/day)."""
    n = len(production_t)
    dates = pd.date_range(start, periods=n, freq="D")
    kwh = BASE_LOAD_KWH + VARIABLE_SEC_KWH_PER_T * production_t + rng.normal(0, noise_kwh, n)
    return pd.DataFrame(
        {
            "period_start": dates,
            "period_end": dates,
            "kwh": kwh,
            "production": production_t,
            "production_unit": "t",
            "operating_days": 1.0,
            "source": "simulated",
            "dataset": "test",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


def wide_plant(rng, production_factor):
    """Baseline 2024 with production 50-150 t/day (mean 100), reporting 2025 at factor x that."""
    baseline = plant_frame(rng, "2024-01-01", rng.uniform(50, 150, 365))
    reporting = plant_frame(rng, "2025-01-01", production_factor * rng.uniform(50, 150, 365))
    return baseline, reporting


def stable_plant(rng):
    """Baseline and reporting both at 90-110 t/day: production barely changes."""
    baseline = plant_frame(rng, "2024-01-01", rng.uniform(90, 110, 365))
    reporting = plant_frame(rng, "2025-01-01", rng.uniform(90, 110, 365))
    return baseline, reporting


def sec_ratio_pct_from_means(mean_base_t, mean_report_t):
    """Analytic SEC-ratio "saving" of the noise-free plant with no retrofit.
    SEC = kWh / t = 2000 / P + 15 at mean production P, so
    1 - SEC_r / SEC_b = 1 - (2000/P_r + 15) / (2000/P_b + 15).
    At P_b = 100 and P_r = 120: 1 - 31.67 / 35 = +9.5% although nothing was retrofitted: the
    fixed 2000 kWh/day is spread over 20% more tonnes. At P_r = 80: 1 - 40 / 35 = -14.3%."""
    sec_base = BASE_LOAD_KWH / mean_base_t + VARIABLE_SEC_KWH_PER_T
    sec_report = BASE_LOAD_KWH / mean_report_t + VARIABLE_SEC_KWH_PER_T
    return 100 * (1 - sec_report / sec_base)


def bill_frame(starts, days_per_bill, kwh_per_day):
    """Billing cycles of a fixed length at a constant daily usage (no noise)."""
    starts = pd.DatetimeIndex(starts)
    n = len(starts)
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + pd.Timedelta(days=days_per_bill - 1),
            "kwh": kwh_per_day * days_per_bill,
            "production": 50.0 * days_per_bill,
            "production_unit": "t",
            "operating_days": float(days_per_bill),
            "source": "utility_bill",
            "dataset": "test",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


def daily_bills(kwh_per_day, production_t, start="2024-01-01", n=10):
    """Tiny daily frame with exact round numbers, for verdict boundaries."""
    return bill_frame(pd.date_range(start, periods=n), 1, kwh_per_day).assign(
        production=production_t, source="simulated"
    )


# ---------- the core claim (validation_plan §1, blind simulator test in miniature) ----------


def test_production_up_20_percent_fakes_a_sec_ratio_saving_saveproof_does_not():
    baseline, reporting = wide_plant(np.random.default_rng(42), production_factor=1.2)
    expected_pct = sec_ratio_pct_from_means(
        baseline["production"].mean(), reporting["production"].mean()
    )
    bee = bee_sec_savings(baseline, reporting)
    assert expected_pct > 5  # ~9.5% at the nominal means; sample means move it by < 1 pp
    # Noise sd 20 kWh/day averaged over 365 days moves SEC by ~0.003%; abs=0.2 pp is ample.
    assert bee.savings_pct == pytest.approx(expected_pct, abs=0.2)
    assert bee.savings_kwh > 0

    ours = verify_savings(fit_baseline(baseline, FORMULA), reporting)
    assert ours.band_low_pct < 0 < ours.band_high_pct
    assert ours.verdict != "verified"


def test_production_down_20_percent_makes_sec_ratio_report_a_loss():
    baseline, reporting = wide_plant(np.random.default_rng(42), production_factor=0.8)
    expected_pct = sec_ratio_pct_from_means(
        baseline["production"].mean(), reporting["production"].mean()
    )
    bee = bee_sec_savings(baseline, reporting)
    assert expected_pct < -10  # ~-14.3% at the nominal means
    assert bee.savings_pct == pytest.approx(expected_pct, abs=0.2)  # same reasoning as above
    assert bee.verdict == "below"

    ours = verify_savings(fit_baseline(baseline, FORMULA), reporting)
    assert ours.band_low_pct < 0 < ours.band_high_pct


def test_naive_reports_negative_savings_when_production_rises():
    """+20% production with no retrofit uses ~8.6% more kWh/day (3800 vs 3500)."""
    baseline, reporting = wide_plant(np.random.default_rng(42), production_factor=1.2)
    naive = naive_savings(baseline, reporting)
    assert naive.savings_pct < -5
    assert naive.savings_kwh < 0
    assert naive.verdict == "below"


# ---------- all three methods agree where production is stable ----------


def test_stable_production_true_15_percent_saving_all_three_methods_agree():
    """Reporting kWh x 0.85. With 90-110 t/day the sample means differ by ~0.3 t (sd), which moves
    naive and SEC-ratio by ~0.15 pp; noise adds ~0.03 pp. 1.5 pp is a 10-sd tolerance that still
    separates 15% from the 10% threshold."""
    baseline, reporting = stable_plant(np.random.default_rng(42))
    reporting = reporting.assign(kwh=reporting["kwh"] * 0.85)
    naive = naive_savings(baseline, reporting)
    bee = bee_sec_savings(baseline, reporting)
    ours = verify_savings(fit_baseline(baseline, FORMULA), reporting)
    for savings_pct in (naive.savings_pct, bee.savings_pct, ours.savings_pct):
        assert savings_pct == pytest.approx(15.0, abs=1.5)
    assert naive.verdict == bee.verdict == "verified"


def test_no_change_at_all_gives_about_zero_for_all_three():
    """Same process in both years; tolerance as in the 15% test (sampling sd ~0.15 pp)."""
    baseline, reporting = stable_plant(np.random.default_rng(42))
    naive = naive_savings(baseline, reporting)
    bee = bee_sec_savings(baseline, reporting)
    ours = verify_savings(fit_baseline(baseline, FORMULA), reporting)
    for savings_pct in (naive.savings_pct, bee.savings_pct, ours.savings_pct):
        assert savings_pct == pytest.approx(0.0, abs=1.5)
    assert naive.verdict == bee.verdict == "below"


# ---------- known answers and invariants ----------


def test_naive_known_answer():
    """Baseline 1000 kWh/day for 10 days, reporting 900 kWh/day for 5 days:
    expected = 1000 x 5 = 5000, savings = 5000 - 4500 = 500 kWh = 10%."""
    baseline = daily_bills(1000.0, 10.0, n=10)
    reporting = daily_bills(900.0, 10.0, start="2025-01-01", n=5)
    result = naive_savings(baseline, reporting)
    assert result.method == "naive"
    assert result.savings_kwh == pytest.approx(500.0, rel=1e-12)
    assert result.savings_pct == pytest.approx(10.0, rel=1e-12)


def test_bee_sec_known_answer():
    """SEC_base = 1000/10 = 100 kWh/t; reporting 900 kWh at 12 t/day -> SEC_report = 75 kWh/t.
    savings = (100 - 75) x 12 t x 5 days = 1500 kWh; savings_pct = 100 x (1 - 75/100) = 25%."""
    baseline = daily_bills(1000.0, 10.0, n=10)
    reporting = daily_bills(900.0, 12.0, start="2025-01-01", n=5)
    result = bee_sec_savings(baseline, reporting)
    assert result.method == "bee_sec"
    assert result.savings_kwh == pytest.approx(1500.0, rel=1e-12)
    assert result.savings_pct == pytest.approx(25.0, rel=1e-12)


def test_unequal_bill_lengths_with_same_daily_usage_give_zero_naive_savings():
    """Baseline bills of 28 days, reporting bills of 31 days, both 400 kWh/day. An unnormalised
    bill comparison would show 31/28 - 1 = 10.7% more use; per-day normalisation shows 0."""
    baseline = bill_frame(pd.date_range("2024-01-01", periods=6, freq="28D"), 28, 400.0)
    reporting = bill_frame(pd.date_range("2025-01-01", periods=6, freq="31D"), 31, 400.0)
    result = naive_savings(baseline, reporting)
    assert result.savings_pct == pytest.approx(0.0, abs=1e-9)
    assert result.savings_kwh == pytest.approx(0.0, abs=1e-6)


def test_scaling_kwh_in_both_periods_scales_savings_kwh_not_pct():
    baseline, reporting = wide_plant(np.random.default_rng(42), production_factor=1.2)
    k = 3.0
    for method in (naive_savings, bee_sec_savings):
        a = method(baseline, reporting)
        b = method(
            baseline.assign(kwh=baseline["kwh"] * k), reporting.assign(kwh=reporting["kwh"] * k)
        )
        assert b.savings_kwh == pytest.approx(k * a.savings_kwh, rel=1e-9)
        assert b.savings_pct == pytest.approx(a.savings_pct, rel=1e-9)


# ---------- the verdict (validation_plan §2: point estimate vs 10%) ----------


@pytest.mark.parametrize(
    ("reporting_kwh_per_day", "expected"),
    [
        (900.0, "verified"),  # exactly 10.0% saving -> verified (>= threshold)
        (900.5, "below"),  # 9.95%
    ],
)
def test_verdict_boundary_at_exactly_10_percent(reporting_kwh_per_day, expected):
    baseline = daily_bills(1000.0, 10.0, n=10)
    reporting = daily_bills(reporting_kwh_per_day, 10.0, start="2025-01-01", n=10)
    for method in (naive_savings, bee_sec_savings):
        result = method(baseline, reporting)
        if expected == "verified":
            assert result.savings_pct == 10.0
        assert result.verdict == expected


# ---------- exclusions: the same rows as verify_savings ----------


def test_absurd_kwh_on_event_and_low_coverage_rows_changes_nothing():
    baseline, reporting = stable_plant(np.random.default_rng(42))
    reporting = reporting.assign(kwh=reporting["kwh"] * 0.85)
    for frame in (baseline, reporting):
        frame.loc[10:19, "event"] = "shutdown"
        frame.loc[40:49, "coverage_pct"] = 50.0
    absurd_baseline, absurd_reporting = baseline.copy(), reporting.copy()
    for frame in (absurd_baseline, absurd_reporting):
        frame.loc[10:19, "kwh"] = 1e9
        frame.loc[40:49, "kwh"] = 1e9
    for method in (naive_savings, bee_sec_savings):
        assert method(absurd_baseline, absurd_reporting) == method(baseline, reporting)


def test_naive_formula_decides_whether_missing_production_rows_are_dropped():
    """Experiments pass the SaveProof formula so naive sees the same rows as verify_savings."""
    baseline, reporting = stable_plant(np.random.default_rng(42))
    reporting.loc[5, "production"] = np.nan
    reporting.loc[5, "kwh"] = 1e9
    without_row = reporting.drop(index=5)
    dropped = naive_savings(baseline, reporting, formula="kwh ~ production")
    assert dropped == naive_savings(baseline, without_row, formula="kwh ~ production")
    kept = naive_savings(baseline, reporting)
    assert kept != naive_savings(baseline, without_row)
    assert kept.savings_pct < -1000  # the 1e9 kWh row was counted


def test_bee_sec_ignores_rows_with_missing_production():
    baseline, reporting = stable_plant(np.random.default_rng(42))
    expected = bee_sec_savings(baseline, reporting)
    reporting.loc[5, "production"] = np.nan
    reporting.loc[5, "kwh"] = 1e9
    reporting_without_row = reporting.drop(index=5)
    assert bee_sec_savings(baseline, reporting) == bee_sec_savings(baseline, reporting_without_row)
    assert bee_sec_savings(baseline, reporting) != expected  # the row really was dropped, not 1e9


# ---------- input checks ----------


def test_different_production_unit_raises():
    baseline, reporting = stable_plant(np.random.default_rng(42))
    with pytest.raises(ValueError, match="production_unit"):
        bee_sec_savings(baseline, reporting.assign(production_unit="pcs"))


def test_zero_total_production_raises():
    baseline, reporting = stable_plant(np.random.default_rng(42))
    with pytest.raises(ValueError, match="production"):
        bee_sec_savings(baseline, reporting.assign(production=0.0))
    with pytest.raises(ValueError, match="production"):
        bee_sec_savings(baseline.assign(production=0.0), reporting)


def test_contract_violation_is_reported():
    baseline, reporting = stable_plant(np.random.default_rng(42))
    reporting.loc[3, "kwh"] = -1.0
    for method in (naive_savings, bee_sec_savings):
        with pytest.raises(ValueError, match="kwh"):
            method(baseline, reporting)


def test_no_usable_rows_raises():
    baseline, reporting = stable_plant(np.random.default_rng(42))
    for method in (naive_savings, bee_sec_savings):
        with pytest.raises(ValueError, match="usable"):
            method(baseline, reporting.assign(event="shutdown"))


# ---------- ComparatorResult ----------


def test_result_is_frozen_and_to_dict_round_trips_through_json():
    baseline, reporting = stable_plant(np.random.default_rng(42))
    for method in (naive_savings, bee_sec_savings):
        result = method(baseline, reporting)
        assert isinstance(result, ComparatorResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.verdict = "verified"
        d = result.to_dict()
        assert json.loads(json.dumps(d)) == d
        assert set(d) == {"method", "savings_kwh", "savings_pct", "verdict"}
        assert all(type(v) in (str, float) for v in d.values())
