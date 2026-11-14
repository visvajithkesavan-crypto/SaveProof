"""Simulator (R3, docs/validation_plan.md §6). Plants are small or noise-free where a hand-computed
answer is needed; rng = np.random.default_rng(seed) with fixed seeds throughout."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from saveproof.contract import REQUIRED, validate
from saveproof.engine.baseline import fit_baseline
from saveproof.engine.savings import verify_savings
from saveproof.ingest.simulator import (
    SimParams,
    SimTruth,
    sample_params,
    simulate_plant,
)

DAILY_FORMULA = "kwh ~ production + C(day_type)"


def params(**overrides) -> SimParams:
    """Daily S1 foundry plant, 1000 kWh/day at 10 t/day, no saving, no shutdown, low noise."""
    base = dict(
        sector="foundry",
        granularity="daily",
        stratum="S1",
        fixed_share=0.3,
        mean_kwh_per_day=1000.0,
        production_mean=10.0,
        production_change_pct=0.0,
        ar1_phi=0.3,
        noise_cv=0.01,
        saving_base_pct=0.0,
        saving_variable_pct=0.0,
        shutdown_days=0,
        misspec="none",
    )
    return SimParams(**{**base, **overrides})


def split(df, meta):
    baseline = df[df["period_start"] <= meta.baseline_end]
    reporting = df[df["period_start"] >= meta.reporting_start]
    return baseline, reporting


# ---------- known answers with the engine ----------


def test_zero_saving_gives_about_zero_within_the_band():
    p = params(noise_cv=0.05, ar1_phi=0.5, shutdown_days=7)
    df, meta, truth = simulate_plant(p, np.random.default_rng(42))
    assert truth.true_savings_kwh == pytest.approx(0.0, abs=1e-6)
    baseline, reporting = split(df, meta)
    result = verify_savings(fit_baseline(baseline, DAILY_FORMULA), reporting)
    assert result.band_low_pct <= 0 <= result.band_high_pct


def test_known_20pct_saving_on_low_noise_s1_plant_is_recovered_within_1_5_pp():
    p = params(saving_base_pct=20.0, saving_variable_pct=20.0)
    df, meta, truth = simulate_plant(p, np.random.default_rng(42))
    # both parts cut 20% and the same noise multiplies both, so the truth is exactly 20%
    assert truth.true_savings_pct == pytest.approx(20.0, rel=1e-12)
    baseline, reporting = split(df, meta)
    result = verify_savings(fit_baseline(baseline, DAILY_FORMULA), reporting)
    assert result.savings_pct == pytest.approx(truth.true_savings_pct, abs=1.5)
    assert result.verdict == "verified"


def test_truth_equals_hand_computed_counterfactual_on_tiny_noise_free_plant():
    """f = 0.25, M = 1000 kWh/day, P = 10 t/day: fixed F = 0.25 x 1000 = 250 kWh/day, variable
    SEC S = 0.75 x 1000 / 10 = 75 kWh/t. Retrofit cuts fixed 20% and variable 10%, so a reporting
    day uses 200 + 67.5 p; without the retrofit (counterfactual) it would use 250 + 75 p at the
    same p; saved = 50 + 7.5 p per day."""
    p = params(
        baseline_days=14,
        reporting_days=14,
        fixed_share=0.25,
        noise_cv=0.0,
        saving_base_pct=20.0,
        saving_variable_pct=10.0,
    )
    df, meta, truth = simulate_plant(p, np.random.default_rng(3))
    baseline, reporting = split(df, meta)
    assert len(baseline) == 14 and len(reporting) == 14
    np.testing.assert_allclose(baseline["kwh"], 250 + 75 * baseline["production"], rtol=1e-12)
    np.testing.assert_allclose(reporting["kwh"], 200 + 67.5 * reporting["production"], rtol=1e-12)
    saved_kwh = (50 + 7.5 * reporting["production"]).sum()
    counterfactual_kwh = (250 + 75 * reporting["production"]).sum()
    assert truth.true_savings_kwh == pytest.approx(saved_kwh, rel=1e-12)
    assert truth.true_savings_pct == pytest.approx(100 * saved_kwh / counterfactual_kwh, rel=1e-12)


def test_fixed_share_is_recovered_by_the_fitted_intercept_on_noise_free_plant():
    p = params(fixed_share=0.2, noise_cv=0.0, shutdown_days=5)
    df, meta, _ = simulate_plant(p, np.random.default_rng(5))
    baseline, _ = split(df, meta)
    model = fit_baseline(baseline, "kwh ~ production")
    # noise-free and linear: OLS is exact up to floating point
    assert model.params["Intercept"] / p.mean_kwh_per_day == pytest.approx(0.2, rel=1e-9)
    assert model.params["production"] == pytest.approx(0.8 * 1000 / 10, rel=1e-9)


# ---------- output format ----------


def test_daily_frame_is_a_valid_contract_table_with_meta():
    df, meta, truth = simulate_plant(params(), np.random.default_rng(1), plant_id="plant_007")
    validate(df)
    assert list(df.columns) == REQUIRED  # nothing else: the engine is not told the physics
    assert len(df) == 730
    assert set(df["source"]) == {"simulated"}
    assert set(df["production_unit"]) == {"t"}
    assert set(df["dataset"]) == {"plant_007"}
    assert meta.plant_id == "plant_007"
    assert meta.timezone == "Asia/Kolkata"
    assert meta.retrofit_date == meta.reporting_start
    assert (meta.reporting_start - meta.baseline_start).days == 365
    assert isinstance(truth, SimTruth)


def test_shutdown_rows_carry_the_event_and_zero_production():
    df, _, _ = simulate_plant(params(shutdown_days=7), np.random.default_rng(2))
    shutdown = df["event"] == "shutdown"
    assert shutdown.sum() == 14  # one 7-day block in each period
    assert (df.loc[shutdown, "production"] == 0).all()
    assert (df.loc[shutdown, "operating_days"] == 0).all()
    assert df.loc[~shutdown, "event"].isna().all()
    assert (df.loc[~shutdown, "operating_days"] == 1).all()


def test_monthly_frame_has_12_plus_12_month_rows_and_validates():
    p = params(granularity="monthly", shutdown_days=10)
    df, meta, _ = simulate_plant(p, np.random.default_rng(4))
    validate(df)
    assert len(df) == 24
    assert df["period_start"].dt.is_month_start.all()
    assert df["period_end"].dt.is_month_end.all()
    assert (df["period_start"] <= meta.baseline_end).sum() == 12
    assert df["operating_days"].sum() == 730 - 20
    assert df["event"].isna().all()  # shutdowns show up as fewer operating_days on a bill


def test_monthly_is_the_daily_plant_summed_by_calendar_month():
    daily, _, truth_daily = simulate_plant(params(shutdown_days=6), np.random.default_rng(9))
    monthly, _, truth_monthly = simulate_plant(
        params(granularity="monthly", shutdown_days=6), np.random.default_rng(9)
    )
    by_month = daily.groupby(daily["period_start"].dt.to_period("M"))[["kwh", "production"]].sum()
    np.testing.assert_allclose(monthly["kwh"], by_month["kwh"], rtol=1e-12)
    np.testing.assert_allclose(monthly["production"], by_month["production"], rtol=1e-12)
    assert truth_monthly == truth_daily


def test_monthly_needs_whole_calendar_months():
    with pytest.raises(ValueError, match="calendar months"):
        simulate_plant(params(granularity="monthly", baseline_days=100), np.random.default_rng(0))


def test_same_seed_gives_identical_plant():
    p = params(noise_cv=0.07, shutdown_days=4, misspec="product_mix", stratum="S4")
    a = simulate_plant(p, np.random.default_rng(11))
    b = simulate_plant(p, np.random.default_rng(11))
    pd.testing.assert_frame_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]


def test_realised_production_change_follows_the_parameter():
    df, meta, _ = simulate_plant(params(production_change_pct=20.0), np.random.default_rng(6))
    baseline, reporting = split(df, meta)
    change_pct = 100 * (reporting["production"].mean() / baseline["production"].mean() - 1)
    # monthly order shocks (sd 8%, 12 per period) leave ~3 pp sd on the realised change: 3 sd
    assert change_pct == pytest.approx(20.0, abs=10)


# ---------- S4 misspecification (same draws as the "none" plant, physics differs) ----------


def test_nonlinear_sec_rises_only_below_mean_throughput():
    none, meta, _ = simulate_plant(params(noise_cv=0.0), np.random.default_rng(8))
    nonlinear, _, _ = simulate_plant(
        params(noise_cv=0.0, misspec="nonlinear", stratum="S4"), np.random.default_rng(8)
    )
    pd.testing.assert_series_equal(none["production"], nonlinear["production"])
    low = none["production"] < 10.0
    assert (nonlinear.loc[low & (none["production"] > 0), "kwh"] > none.loc[low, "kwh"]).all()
    np.testing.assert_allclose(nonlinear.loc[~low, "kwh"], none.loc[~low, "kwh"], rtol=1e-12)


def test_product_mix_changes_reporting_sec_only():
    none, meta, _ = simulate_plant(params(noise_cv=0.0), np.random.default_rng(8))
    mix, _, _ = simulate_plant(
        params(noise_cv=0.0, misspec="product_mix", stratum="S4"), np.random.default_rng(8)
    )
    base = none["period_start"] <= meta.baseline_end
    np.testing.assert_allclose(mix.loc[base, "kwh"], none.loc[base, "kwh"], rtol=1e-12)
    fixed_kwh = 0.3 * 1000
    running = ~base & (none["production"] > 0)
    ratio = (mix.loc[running, "kwh"] - fixed_kwh) / (none.loc[running, "kwh"] - fixed_kwh)
    assert np.ptp(ratio) < 1e-9  # one SEC factor for the whole reporting period
    assert abs(ratio.iloc[0] - 1) > 0.01


def test_params_reject_unknown_labels():
    with pytest.raises(ValueError, match="granularity"):
        params(granularity="weekly")
    with pytest.raises(ValueError, match="misspec"):
        params(misspec="weather")
    with pytest.raises(ValueError, match="stratum"):
        params(stratum="S5")


# ---------- sample_params (§6 composition) ----------

CHANGE_RANGES = {"S1": (-5, 5), "S2": (10, 30), "S3": (-30, -10)}


@pytest.mark.parametrize("stratum", ["S1", "S2", "S3", "S4"])
def test_sampled_production_change_lands_in_the_stratum_range(stratum):
    rng = np.random.default_rng(12)
    for _ in range(200):
        p = sample_params(stratum, "daily", "foundry", rng)
        assert p.stratum == stratum
        if stratum == "S4":
            assert 10 <= abs(p.production_change_pct) <= 30
            assert p.misspec in ("product_mix", "nonlinear")
        else:
            low, high = CHANGE_RANGES[stratum]
            assert low <= p.production_change_pct <= high
            assert p.misspec == "none"


def test_sampled_noise_saving_and_fixed_share_ranges():
    rng = np.random.default_rng(13)
    samples = [
        sample_params("S1", g, s, rng)
        for _ in range(300)
        for g in ("daily", "monthly")
        for s in ("foundry", "dairy_cold_chain")
    ]
    for p in samples:
        assert 0.2 <= p.ar1_phi <= 0.7
        assert 0.03 <= p.noise_cv <= 0.10
        assert 0 <= p.saving_base_pct <= 100 and 0 <= p.saving_variable_pct <= 100
        # saving as a share of baseline-mix energy: fixed part x base cut + variable x its cut
        total_pct = p.fixed_share * p.saving_base_pct + (1 - p.fixed_share) * p.saving_variable_pct
        assert 0 <= total_pct <= 20 + 1e-9
        band = (0.10, 0.30) if p.sector == "foundry" else (0.30, 0.50)
        assert band[0] <= p.fixed_share <= band[1]
    shutdown_share = np.mean([p.shutdown_days > 0 for p in samples])
    assert 0.15 <= shutdown_share <= 0.25  # ~20% of plants; binomial sd ~1% at n = 1200


def test_sample_params_is_reproducible_and_rejects_unknown_sector():
    a = sample_params("S2", "monthly", "dairy_cold_chain", np.random.default_rng(1))
    b = sample_params("S2", "monthly", "dairy_cold_chain", np.random.default_rng(1))
    assert a == b
    assert dataclasses.asdict(a)["granularity"] == "monthly"
    with pytest.raises(ValueError, match="sector"):
        sample_params("S1", "daily", "textile", np.random.default_rng(1))
