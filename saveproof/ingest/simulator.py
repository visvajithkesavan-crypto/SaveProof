"""Plant simulator for the blind test (docs/validation_plan.md §6).

A plant is simulated day by day: kWh = (fixed load + variable SEC x production) x (1 + AR(1)
noise), with a retrofit at the start of the reporting period. The output is the contract table
(source "simulated"); the true saving is returned separately as SimTruth and never appears in the
table, and S4 misspecification is never visible in it. Dates are plant-local (Asia/Kolkata, no
DST) midnights; a daily row covers one day, a monthly row a whole calendar month (ends inclusive).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from saveproof.contract import PlantMeta, validate

# ---------- blind-test design. Source: docs/validation_plan.md §6 ----------
STRATA = ("S1", "S2", "S3", "S4")
GRANULARITIES = ("daily", "monthly")
MISSPECS = ("none", "product_mix", "nonlinear")
# Production change baseline -> reporting, percent, per stratum (S4: magnitude, random sign).
S1_PRODUCTION_CHANGE_PCT = (-5.0, 5.0)
S2_PRODUCTION_CHANGE_PCT = (10.0, 30.0)
S3_PRODUCTION_CHANGE_PCT = (-30.0, -10.0)
S4_PRODUCTION_CHANGE_ABS_PCT = (10.0, 30.0)
# Target true saving, percent of baseline-mix energy, uniform; split randomly into base/variable.
TRUE_SAVING_PCT = (0.0, 20.0)
AR1_PHI = (0.2, 0.7)  # lag-1 autocorrelation of daily kWh noise, uniform
NOISE_CV = (0.03, 0.10)  # standard deviation of daily kWh noise, fraction, uniform
SHUTDOWN_PLANT_SHARE = 0.20  # share of plants with shutdown days, fraction

# ---------- fixed share: output-independent fraction of baseline kWh, uniform by sector ----------
# Source: UNIDO EE&RE Technology Compendium, Foundry cluster Ahmedabad pp. 45, 59, 68, 90, 117 and
# Coimbatore p. 20. Inferred from components, not stated directly.
FOUNDRY_FIXED_SHARE = (0.10, 0.30)
# Source: UNIDO Technology Compendium, Gujarat Dairy cluster p. 39. Inferred from components, not
# stated directly.
DAIRY_FIXED_SHARE = (0.30, 0.50)
FIXED_SHARE = {"foundry": FOUNDRY_FIXED_SHARE, "dairy_cold_chain": DAIRY_FIXED_SHARE}

# ---------- SaveProof assumptions with no source: TODO(verify) ----------
# Scale only: savings %, errors and verdicts do not depend on it (every kWh term scales with
# MEAN_KWH_PER_DAY, and production rescales the SEC). kWh/day and t/day per sector.
MEAN_KWH_PER_DAY = {"foundry": 8000.0, "dairy_cold_chain": 6000.0}
PRODUCTION_MEAN_T_PER_DAY = {"foundry": 10.0, "dairy_cold_chain": 50.0}
# First simulated day: start of an Indian financial year, so 365 + 365 days are whole months.
SIM_START = pd.Timestamp("2025-04-01")
WEEKEND_PRODUCTION_FACTOR = 0.5  # Saturday/Sunday production / weekday production
PRODUCTION_DAILY_CV = 0.10  # day-to-day production variation, fraction
PRODUCTION_MONTHLY_CV = 0.08  # month-to-month (order book) production variation, fraction
SHUTDOWN_DAYS = (3, 14)  # length of the one shutdown block in each period, days, uniform integer
# Non-linear SEC: variable SEC x (1 + gain x (1 - p / production_mean)) when p < production_mean.
NONLINEAR_SEC_GAIN = 0.5
# Product mix: product B's SEC / product A's, B's baseline share of output, and the absolute
# change in B's share in the reporting period (uniform magnitude, random sign).
PRODUCT_MIX_SEC_RATIO = 1.5
PRODUCT_MIX_BASE_SHARE = 0.3
PRODUCT_MIX_SHIFT = (0.1, 0.2)

TIMEZONE = "Asia/Kolkata"


@dataclass(frozen=True, kw_only=True)
class SimParams:
    """One plant. Percent fields are 0-100; fixed_share and noise_cv are fractions."""

    sector: str  # "foundry" | "dairy_cold_chain"
    granularity: str  # "daily" | "monthly"
    stratum: str  # "S1".."S4"
    baseline_days: int = 365
    reporting_days: int = 365
    fixed_share: float  # share of baseline mean kWh/day that does not scale with output
    mean_kwh_per_day: float  # expected baseline kWh per calendar day
    production_mean: float  # expected baseline production, t per calendar day
    production_change_pct: float  # expected production change, baseline -> reporting
    ar1_phi: float
    noise_cv: float
    saving_base_pct: float  # retrofit cut of the fixed load
    saving_variable_pct: float  # retrofit cut of the variable SEC
    shutdown_days: int  # length of one shutdown block in each period; 0 for none
    misspec: str  # "none" | "product_mix" | "nonlinear"

    def __post_init__(self) -> None:
        for name, allowed in (
            ("granularity", GRANULARITIES),
            ("stratum", STRATA),
            ("misspec", MISSPECS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} is {getattr(self, name)!r}, expected one of {allowed}")
        if not 0 <= self.fixed_share < 1:
            raise ValueError(f"fixed_share must be in [0, 1), got {self.fixed_share}")
        for name in ("saving_base_pct", "saving_variable_pct"):
            if not 0 <= getattr(self, name) <= 100:
                raise ValueError(f"{name} must be in [0, 100], got {getattr(self, name)}")
        if not -1 < self.ar1_phi < 1:
            raise ValueError(f"ar1_phi must be in (-1, 1), got {self.ar1_phi}")
        if not 0 <= self.shutdown_days < min(self.baseline_days, self.reporting_days):
            raise ValueError(f"shutdown_days must fit in each period, got {self.shutdown_days}")


@dataclass(frozen=True)
class SimTruth:
    """Saving against the counterfactual: the same plant in the reporting period without the
    retrofit, at its actual production (and product mix) and with the same noise draws. Summed
    over running days (shutdown days use 0 kWh either way)."""

    true_savings_kwh: float
    true_savings_pct: float  # 100 x saved / counterfactual kWh


def _ar1_noise(n: int, phi: float, cv: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) series with standard deviation cv (fraction of kWh)."""
    noise = np.empty(n)
    noise[0] = rng.normal(0.0, cv)
    shocks = rng.normal(0.0, cv * np.sqrt(1 - phi**2), n)
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + shocks[t]
    return noise


def _shutdown_mask(params: SimParams, rng: np.random.Generator) -> np.ndarray:
    """True on shutdown days: one contiguous block of shutdown_days in each period."""
    mask = np.zeros(params.baseline_days + params.reporting_days, dtype=bool)
    if params.shutdown_days == 0:
        return mask
    for start, length in ((0, params.baseline_days), (params.baseline_days, params.reporting_days)):
        first = start + int(rng.integers(0, length - params.shutdown_days + 1))
        mask[first : first + params.shutdown_days] = True
    return mask


def _production_t(
    params: SimParams, days: pd.DatetimeIndex, reporting: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Production per day before shutdowns (t). Weekends run at WEEKEND_PRODUCTION_FACTOR; the
    weekday level is set so the expected baseline mean is production_mean."""
    day_factor = np.where(days.dayofweek >= 5, WEEKEND_PRODUCTION_FACTOR, 1.0)
    weekday_t = params.production_mean / day_factor[~reporting].mean()
    level = np.where(reporting, 1 + params.production_change_pct / 100, 1.0)
    month_codes = pd.factorize(days.to_period("M"))[0]
    month_shock = rng.normal(0.0, PRODUCTION_MONTHLY_CV, month_codes.max() + 1)[month_codes]
    shock = 1 + month_shock + rng.normal(0.0, PRODUCTION_DAILY_CV, len(days))
    if (shock <= 0).any():
        raise ValueError("production shock <= -100% drawn; production would be negative")
    return weekday_t * day_factor * level * shock


def _sec_factor(
    params: SimParams, production_t: np.ndarray, reporting: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Daily multiplier on the variable SEC: 1 unless misspec (S4), which the table never shows."""
    if params.misspec == "nonlinear":
        shortfall = np.maximum(0.0, 1 - production_t / params.production_mean)
        return 1 + NONLINEAR_SEC_GAIN * shortfall
    if params.misspec == "product_mix":
        shift = rng.uniform(*PRODUCT_MIX_SHIFT) * rng.choice([-1.0, 1.0])
        share_b = np.where(reporting, PRODUCT_MIX_BASE_SHARE + shift, PRODUCT_MIX_BASE_SHARE)
        baseline_mix = 1 - PRODUCT_MIX_BASE_SHARE + PRODUCT_MIX_BASE_SHARE * PRODUCT_MIX_SEC_RATIO
        return (1 - share_b + share_b * PRODUCT_MIX_SEC_RATIO) / baseline_mix
    return np.ones(len(production_t))


def _to_months(daily: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month bills: kWh, production and operating_days summed; event left empty
    (shutdowns show as fewer operating_days). Raises unless every month is whole."""
    months = daily.groupby(daily["period_start"].dt.to_period("M"))
    out = months.agg(
        period_start=("period_start", "min"),
        period_end=("period_end", "max"),
        kwh=("kwh", "sum"),
        production=("production", "sum"),
        operating_days=("operating_days", "sum"),
    ).reset_index(drop=True)
    whole = out["period_start"].dt.is_month_start & out["period_end"].dt.is_month_end
    if not whole.all():
        raise ValueError(
            "monthly granularity needs whole calendar months in both periods; "
            f"month starting {out.loc[~whole, 'period_start'].iloc[0].date()} is partial"
        )
    for column in ("production_unit", "source", "dataset", "coverage_pct"):
        out[column] = daily[column].iloc[0]
    out["event"] = pd.Series([None] * len(out), dtype=object)
    return out[daily.columns]


def simulate_plant(
    params: SimParams, rng: np.random.Generator, plant_id: str = "sim_plant"
) -> tuple[pd.DataFrame, PlantMeta, SimTruth]:
    """Simulate one plant: contract table (daily or monthly, validated), PlantMeta (retrofit on
    the first reporting day) and SimTruth. kWh per period, production in t."""
    n_days = params.baseline_days + params.reporting_days
    days = pd.date_range(SIM_START, periods=n_days)
    reporting = np.arange(n_days) >= params.baseline_days
    running = ~_shutdown_mask(params, rng)
    production_t = _production_t(params, days, reporting, rng) * running
    noise = 1 + _ar1_noise(n_days, params.ar1_phi, params.noise_cv, rng)
    sec_kwh_per_t = (1 - params.fixed_share) * params.mean_kwh_per_day / params.production_mean
    variable_kwh = sec_kwh_per_t * production_t * _sec_factor(params, production_t, reporting, rng)
    # A shut-down plant draws no base load (mv-engine skill: base_load x running_flag).
    fixed_kwh = params.fixed_share * params.mean_kwh_per_day * running

    counterfactual_kwh = (fixed_kwh + variable_kwh) * noise
    retrofit_kwh = (
        fixed_kwh * (1 - params.saving_base_pct / 100)
        + variable_kwh * (1 - params.saving_variable_pct / 100)
    ) * noise
    actual_kwh = np.where(reporting, retrofit_kwh, counterfactual_kwh)
    saved_kwh = float((counterfactual_kwh - actual_kwh)[reporting].sum())
    truth = SimTruth(saved_kwh, 100 * saved_kwh / float(counterfactual_kwh[reporting].sum()))

    daily = pd.DataFrame(
        {
            "period_start": days,
            "period_end": days,
            "kwh": actual_kwh,
            "production": production_t,
            "production_unit": "t",
            "operating_days": running.astype(float),
            "source": "simulated",
            "dataset": plant_id,
            "coverage_pct": 100.0,
            "event": pd.Series(np.where(running, None, "shutdown"), dtype=object),
        }
    )
    frame = daily if params.granularity == "daily" else _to_months(daily)
    reporting_start = days[params.baseline_days]
    meta = PlantMeta(
        plant_id=plant_id,
        sector=params.sector,
        timezone=TIMEZONE,
        baseline_start=days[0],
        baseline_end=days[params.baseline_days - 1],
        reporting_start=reporting_start,
        reporting_end=days[-1],
        retrofit_date=reporting_start,
    )
    return validate(frame), meta, truth


def _production_change_pct(stratum: str, rng: np.random.Generator) -> float:
    ranges = {"S1": S1_PRODUCTION_CHANGE_PCT, "S2": S2_PRODUCTION_CHANGE_PCT,
              "S3": S3_PRODUCTION_CHANGE_PCT}  # fmt: skip
    if stratum == "S4":
        return float(rng.uniform(*S4_PRODUCTION_CHANGE_ABS_PCT) * rng.choice([-1.0, 1.0]))
    return float(rng.uniform(*ranges[stratum]))


def sample_params(
    stratum: str, granularity: str, sector: str, rng: np.random.Generator
) -> SimParams:
    """Draw one plant's SimParams per docs/validation_plan.md §6. The target saving T (uniform
    TRUE_SAVING_PCT) is split at a random share u into the base and variable cuts, with
    fixed_share x base + (1 - fixed_share) x variable = T; u is capped so the base cut is at
    most 100%. The realised true saving (SimTruth) also moves with the reporting production."""
    if sector not in FIXED_SHARE:
        raise ValueError(f"sector is {sector!r}, expected one of {sorted(FIXED_SHARE)}")
    if stratum not in STRATA:
        raise ValueError(f"stratum is {stratum!r}, expected one of {STRATA}")
    production_change_pct = _production_change_pct(stratum, rng)
    misspec = str(rng.choice(["product_mix", "nonlinear"])) if stratum == "S4" else "none"
    fixed_share = float(rng.uniform(*FIXED_SHARE[sector]))
    target_pct = float(rng.uniform(*TRUE_SAVING_PCT))
    base_part = float(rng.uniform(0.0, min(1.0, 100 * fixed_share / max(target_pct, 1e-12))))
    has_shutdown = rng.random() < SHUTDOWN_PLANT_SHARE
    shutdown_days = int(rng.integers(SHUTDOWN_DAYS[0], SHUTDOWN_DAYS[1] + 1)) if has_shutdown else 0
    return SimParams(
        sector=sector,
        granularity=granularity,
        stratum=stratum,
        fixed_share=fixed_share,
        mean_kwh_per_day=MEAN_KWH_PER_DAY[sector],
        production_mean=PRODUCTION_MEAN_T_PER_DAY[sector],
        production_change_pct=production_change_pct,
        ar1_phi=float(rng.uniform(*AR1_PHI)),
        noise_cv=float(rng.uniform(*NOISE_CV)),
        saving_base_pct=base_part * target_pct / fixed_share,
        saving_variable_pct=(1 - base_part) * target_pct / (1 - fixed_share),
        shutdown_days=shutdown_days,
        misspec=misspec,
    )
