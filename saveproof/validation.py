"""Validation experiments as pure functions: model specs, null-test windows, injected savings and
the pre-registered pass criteria. The authority is docs/validation_plan.md §3 (specs), §4 (null
test), §5 (injected saving) and §8 (I2 amendment); every constant here cites it. The V5b section
at the end follows docs/validation_addendum_v5b.md and leaves the V5 functions unchanged.

Frames are daily contract tables with plant-local dates; windows are selected on period_start
(the day itself, since daily rows have period_start == period_end). No file I/O: the scripts in
experiments/ load data and write results.
"""

from dataclasses import dataclass, fields

import pandas as pd

from saveproof.contract import validate
from saveproof.engine.baseline import (
    InsufficientDataError,
    add_calendar_terms,
    fit_baseline,
    usable_rows,
)
from saveproof.engine.comparators import bee_sec_savings, naive_savings
from saveproof.engine.savings import SavingsResult, verify_savings
from saveproof.engine.stability import baseline_stability

# Spec name -> (contract `dataset` value, formula). Source: docs/validation_plan.md §3.
SPECS = {
    "UCI-A": ("uci_steel", "kwh ~ C(day_type)"),
    "UCI-B": ("uci_steel", "kwh ~ production + C(day_type)"),
    "IR-A": ("irish_mi_a", "kwh ~ C(day_type)"),
    "IR-B": ("irish_mi_a", "kwh ~ production + C(day_type)"),
}
# Calendar-only specs (original I2 wording) and production-proxy specs (N2 flag, I3).
# Source: docs/validation_plan.md §3, §4 N2, §5 I3, §8.
CALENDAR_SPECS = ("UCI-A", "IR-A")
PROXY_SPECS = ("UCI-B", "IR-B")

# Fixed split. UCI: baseline Jan-Aug 2018, reporting Sep-Dec 2018; Irish: first 60% of days.
# Source: docs/validation_plan.md §4 "Fixed split".
UCI_FIXED_BASELINE = (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-08-31"))
UCI_FIXED_REPORTING = (pd.Timestamp("2018-09-01"), pd.Timestamp("2018-12-31"))
IRISH_BASELINE_SHARE = 0.6
# Rolling windows: dataset -> (baseline days, reporting days), start shifted ROLLING_STEP_DAYS.
# Source: docs/validation_plan.md §4 "Rolling".
ROLLING_DAYS = {"uci_steel": (180, 90), "irish_mi_a": (90, 45)}
ROLLING_STEP_DAYS = 7

# Injected savings in percent and the two shapes. Source: docs/validation_plan.md §5.
INJECTED_SAVING_PCTS = (0, 5, 7, 10, 12, 15)
SHAPES = ("uniform", "weekday_only")
I1_SAVING_PCTS = (0, 5, 7)  # §5 I1: zero verified at these
I3_SAVING_PCTS = (10, 12, 15)  # §5 I3: verified rate (power) reported at these
# N2 flags an adequate spec whose share of null windows with 0 inside the band is below this.
# Unit: fraction. Source: docs/validation_plan.md §4 N2.
N2_FLAG_SHARE = 0.5

_MODEL_STATS = ("cvrmse", "nmbe", "r2", "rho", "n_eff", "p")
_RESULT_KEYS = (
    *(f.name for f in fields(SavingsResult)),
    *_MODEL_STATS,
    "contains_zero",
    "status",
    "status_detail",
)
_STATUSES = ("ok", "insufficient_data")


@dataclass(frozen=True, eq=False)
class Window:
    """One baseline/reporting split. Bounds are plant-local dates, both ends inclusive; the
    frames hold the rows whose period_start falls inside them (a missing date is just absent)."""

    name: str  # "fixed" or "rolling_NN"
    baseline_start: pd.Timestamp
    baseline_end: pd.Timestamp
    reporting_start: pd.Timestamp
    reporting_end: pd.Timestamp
    baseline: pd.DataFrame
    reporting: pd.DataFrame


def _window(name: str, df: pd.DataFrame, baseline: tuple, reporting: tuple) -> Window:
    def rows(bounds: tuple) -> pd.DataFrame:
        return df[df["period_start"].between(*bounds)]

    return Window(name, *baseline, *reporting, rows(baseline), rows(reporting))


def fixed_split(df: pd.DataFrame, dataset: str) -> Window:
    """Fixed null-test split of a daily contract frame (§4): calendar dates for "uci_steel", the
    first round(0.6 * rows) rows as baseline for "irish_mi_a". Raises ValueError if `dataset` is
    unknown or differs from the frame's `dataset` column."""
    if dataset not in ROLLING_DAYS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {sorted(ROLLING_DAYS)}")
    df = validate(df)
    found = sorted(set(df["dataset"]))
    if found != [dataset]:
        raise ValueError(f"column 'dataset' holds {found}, expected only {dataset!r}")
    if dataset == "uci_steel":
        return _window("fixed", df, UCI_FIXED_BASELINE, UCI_FIXED_REPORTING)
    k = round(IRISH_BASELINE_SHARE * len(df))
    dates = df["period_start"]
    return _window("fixed", df, (dates.iloc[0], dates.iloc[k - 1]), (dates.iloc[k], dates.iloc[-1]))


def rolling_windows(
    df: pd.DataFrame, baseline_days: int, reporting_days: int, step_days: int = ROLLING_STEP_DAYS
) -> list[Window]:
    """Calendar windows (§4) starting at the frame's first date and shifted step_days each time;
    only windows whose reporting end is on or before the last date are kept. Day counts are
    calendar days, so a missing date shortens a window's frame but not its bounds."""
    df = validate(df)
    first, last = df["period_start"].iloc[0], df["period_start"].iloc[-1]
    windows = []
    while True:
        baseline_start = first + pd.Timedelta(days=step_days * len(windows))
        baseline_end = baseline_start + pd.Timedelta(days=baseline_days - 1)
        reporting_start = baseline_end + pd.Timedelta(days=1)
        reporting_end = reporting_start + pd.Timedelta(days=reporting_days - 1)
        if reporting_end > last:
            return windows
        name = f"rolling_{len(windows):02d}"
        windows.append(
            _window(name, df, (baseline_start, baseline_end), (reporting_start, reporting_end))
        )


def run_window(baseline: pd.DataFrame, reporting: pd.DataFrame, formula: str) -> dict:
    """Fit `formula` on baseline, verify the reporting frame; one flat results row.

    Keys: every SavingsResult field (gate_failures joined with "; "), the model's cvrmse, nmbe,
    r2, rho, n_eff and p, contains_zero (0 inside the band), status and status_detail. When
    fit_baseline or verify_savings raises InsufficientDataError, status is "insufficient_data",
    status_detail is the message and the unavailable values are None. Other errors propagate.
    """
    row = dict.fromkeys(_RESULT_KEYS)
    try:
        model = fit_baseline(baseline, formula)
        row.update({name: getattr(model, name) for name in _MODEL_STATS})
        result = verify_savings(model, reporting)
    except InsufficientDataError as error:
        return {**row, "status": "insufficient_data", "status_detail": str(error)}
    row.update(result.to_dict())
    row["gate_failures"] = "; ".join(result.gate_failures)
    row["contains_zero"] = bool(result.band_low_pct <= 0 <= result.band_high_pct)
    row["status"] = "ok"
    return row


def inject_saving(reporting: pd.DataFrame, formula: str, s: float, shape: str) -> pd.DataFrame:
    """Copy of the reporting frame with kWh cut by fraction s (§5); production and dates untouched.

    "uniform": every row x (1 - s). "weekday_only": only weekday rows, all by one factor chosen so
    the kWh total over usable_rows(formula) drops by exactly s. Raises ValueError for an unknown
    shape, s outside [0, 1), or weekdays too small to carry the cut.
    """
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    if not 0 <= s < 1:
        raise ValueError(f"s must be a fraction in [0, 1), got {s}")
    out = reporting.copy()
    if shape == "uniform":
        out["kwh"] = out["kwh"] * (1 - s)
        return out

    calendar = add_calendar_terms(reporting)
    usable = usable_rows(calendar, formula)
    weekday = calendar["day_type"] == "weekday"
    total_kwh = reporting.loc[usable, "kwh"].sum()
    weekday_kwh = reporting.loc[usable & weekday, "kwh"].sum()
    if weekday_kwh <= 0:
        raise ValueError("no usable weekday kWh to cut for shape 'weekday_only'")
    # f * weekday + weekend = (1 - s) * (weekday + weekend)  =>  f = 1 - s * total / weekday
    factor = 1 - s * total_kwh / weekday_kwh
    if factor < 0:
        raise ValueError(
            f"cutting {s:.0%} of the usable total needs weekday factor {factor:.3f} < 0"
        )
    out.loc[weekday, "kwh"] = out.loc[weekday, "kwh"] * factor
    return out


# ---------- pre-registered criteria over results tables (one row per run_window call) ----------


def _scored(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows with status "ok", and how many were "insufficient_data" (excluded, but counted)."""
    unknown = ~table["status"].isin(_STATUSES)
    if unknown.any():
        row = table.index[unknown.to_numpy()][0]
        status = table.at[row, "status"]
        raise ValueError(f"column 'status': row {row} is {status!r}, expected one of {_STATUSES}")
    ok = table["status"] == "ok"
    return table[ok], int((~ok).sum())


def _no_verified(table: pd.DataFrame) -> dict:
    scored, n_insufficient = _scored(table)
    n_verified = int((scored["verdict"] == "verified").sum())
    return {
        "pass": n_verified == 0,
        "n_verified": n_verified,
        "n_scored": len(scored),
        "n_insufficient": n_insufficient,
    }


def n1(null_results: pd.DataFrame) -> dict:
    """§4 N1 (hard): zero "verified" in every null window for every spec."""
    return _no_verified(null_results)


def n2(null_results: pd.DataFrame) -> dict:
    """§4 N2 (reported): per spec, share of scored windows whose band contains 0 (None if none
    scored); flagged if a proxy spec (UCI-B, IR-B) is below N2_FLAG_SHARE."""
    out = {}
    for spec, group in null_results.groupby("spec", sort=False):
        scored, n_insufficient = _scored(group)
        share = float(scored["contains_zero"].astype(bool).mean()) if len(scored) else None
        out[spec] = {
            "share_contains_zero": share,
            "n_scored": len(scored),
            "n_insufficient": n_insufficient,
            "flagged": bool(spec in PROXY_SPECS and share is not None and share < N2_FLAG_SHARE),
        }
    return out


def i1(injected_results: pd.DataFrame) -> dict:
    """§5 I1 (hard): zero "verified" at s in I1_SAVING_PCTS, both shapes, every spec."""
    low = injected_results["injected_saving_pct"].isin(I1_SAVING_PCTS)
    return _no_verified(injected_results[low])


def i2_amended(injected_results: pd.DataFrame) -> dict:
    """§5 I2 as amended in §8 (hard): zero "verified" in any run whose adequacy gate failed
    (ashrae_ok False), at every s. Insufficient-data runs are counted, not scored."""
    return _no_verified(injected_results[~injected_results["ashrae_ok"].eq(True)])


def i2_original(injected_results: pd.DataFrame) -> dict:
    """§5 I2 as signed off 2026-11-09, reported alongside the amendment: UCI-A and IR-A give
    zero "verified" at every s."""
    return _no_verified(injected_results[injected_results["spec"].isin(CALENDAR_SPECS)])


def i3(injected_results: pd.DataFrame) -> dict:
    """§5 I3 (reported): verified rate at s in I3_SAVING_PCTS, as {spec: {shape: {"10": {...}}}};
    verified_rate is None where nothing was scored."""
    out = {}
    for spec in injected_results["spec"].unique():
        out[spec] = {}
        for shape in injected_results["shape"].unique():
            out[spec][shape] = {}
            for pct in I3_SAVING_PCTS:
                cell = injected_results[
                    (injected_results["spec"] == spec)
                    & (injected_results["shape"] == shape)
                    & (injected_results["injected_saving_pct"] == pct)
                ]
                counts = _no_verified(cell)
                rate = counts["n_verified"] / counts["n_scored"] if counts["n_scored"] else None
                out[spec][shape][str(pct)] = {
                    "verified_rate": rate,
                    "n_scored": counts["n_scored"],
                    "n_insufficient": counts["n_insufficient"],
                }
    return out


# ---------- V5b: Laayati mine, real production (docs/validation_addendum_v5b.md) ----------

# One formula for all three stations. Source: addendum §5.
V5B_FORMULA = "kwh ~ production"
# The 32 days of the data set, plant-local. Source: addendum §3.
V5B_DAYS = pd.date_range("2018-12-31", "2019-01-31")
# Fixed split: baseline days 1-20, reporting days 21-32. Source: addendum §6.
V5B_FIXED_BASELINE = (pd.Timestamp("2018-12-31"), pd.Timestamp("2019-01-19"))
V5B_FIXED_REPORTING = (pd.Timestamp("2019-01-20"), pd.Timestamp("2019-01-31"))
# Rolling: 14-day baseline, 7-day reporting, step 1 day -> 12 windows. Source: addendum §6.
V5B_ROLLING_DAYS = (14, 7)
V5B_ROLLING_STEP_DAYS = 1
# Injected savings in percent, uniform shape only. Source: addendum §7.
V5B_SAVING_PCTS = (0, 5, 7, 10, 12, 15)
V5B_SHAPE = "uniform"
# B2: zero verified at these; B6: comparator false verified at the low set, power at the high set.
# Source: addendum §8.
B2_SAVING_PCTS = (5, 7)
B6_LOW_SAVING_PCTS = (0, 5, 7)
B6_POWER_SAVING_PCTS = (10, 12, 15)
METHODS = ("saveproof", "bee_sec", "naive")
_SAVINGS_COLUMNS = {
    "saveproof": "savings_pct",
    "bee_sec": "bee_sec_savings_pct",
    "naive": "naive_savings_pct",
}
VERDICT_COLUMNS = {"saveproof": "verdict", "bee_sec": "bee_sec_verdict", "naive": "naive_verdict"}
_STABILITY_KEYS = ("slope_kwh_per_day", "p_value", "drift_share", "flagged")


def v5b_windows(df: pd.DataFrame) -> list[Window]:
    """Fixed split, then the 12 rolling windows (§6), of one station's daily frame. Raises
    ValueError unless the frame holds exactly the 32 days 2018-12-31 to 2019-01-31."""
    df = validate(df)
    if len(df) != len(V5B_DAYS) or not (df["period_start"].to_numpy() == V5B_DAYS).all():
        raise ValueError(
            f"V5b needs the {len(V5B_DAYS)} days 2018-12-31 to 2019-01-31 (addendum §3, §6); "
            f"frame has {len(df)} rows from {df['period_start'].iloc[0].date()} "
            f"to {df['period_start'].iloc[-1].date()}"
        )
    fixed = _window("fixed", df, V5B_FIXED_BASELINE, V5B_FIXED_REPORTING)
    return [fixed, *rolling_windows(df, *V5B_ROLLING_DAYS, step_days=V5B_ROLLING_STEP_DAYS)]


def window_stability(baseline: pd.DataFrame, formula: str) -> dict:
    """baseline_stability (§10) of the model fitted on `baseline`, as stability_* columns; all
    None when fit_baseline raises InsufficientDataError."""
    try:
        result = baseline_stability(fit_baseline(baseline, formula)).to_dict()
    except InsufficientDataError:
        result = dict.fromkeys(_STABILITY_KEYS)
    return {f"stability_{key}": result[key] for key in _STABILITY_KEYS}


def compare_methods(baseline: pd.DataFrame, reporting: pd.DataFrame, formula: str) -> dict:
    """Naive and BEE SEC-ratio on the same frames as SaveProof (§7); naive selects rows with
    `formula` so all three methods drop identical rows."""
    naive = naive_savings(baseline, reporting, formula)
    bee = bee_sec_savings(baseline, reporting)
    return {
        "naive_savings_pct": naive.savings_pct,
        "naive_verdict": naive.verdict,
        "bee_sec_savings_pct": bee.savings_pct,
        "bee_sec_verdict": bee.verdict,
    }


def run_v5b_window(baseline: pd.DataFrame, reporting: pd.DataFrame, formula: str) -> dict:
    """run_window plus gate_passed (adequacy, extrapolation and proxy gates all passed; None when
    insufficient data) and the comparator columns of compare_methods."""
    row = run_window(baseline, reporting, formula)
    row["gate_passed"] = row["gate_failures"] == "" if row["status"] == "ok" else None
    return {**row, **compare_methods(baseline, reporting, formula)}


def _count_verified(table: pd.DataFrame, method: str) -> dict:
    """Like _no_verified, for any method's verdict column."""
    scored, n_insufficient = _scored(table)
    n_verified = int((scored[VERDICT_COLUMNS[method]] == "verified").sum())
    return {
        "pass": n_verified == 0,
        "n_verified": n_verified,
        "n_scored": len(scored),
        "n_insufficient": n_insufficient,
    }


def _null_runs(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["injected_saving_pct"] == 0]


def b1(results: pd.DataFrame) -> dict:
    """§8 B1 (hard): SaveProof gives zero "verified" in all null runs (s = 0)."""
    return _count_verified(_null_runs(results), "saveproof")


def b2(results: pd.DataFrame) -> dict:
    """§8 B2 (hard): SaveProof gives zero "verified" at s in B2_SAVING_PCTS."""
    return _count_verified(
        results[results["injected_saving_pct"].isin(B2_SAVING_PCTS)], "saveproof"
    )


def b3(results: pd.DataFrame) -> dict:
    """§8 B3 (hard): no "verified" in any run where the gate failed (gate_passed not True).
    Insufficient-data runs are counted, not scored."""
    return _count_verified(results[~results["gate_passed"].eq(True)], "saveproof")


def b4(results: pd.DataFrame) -> dict:
    """§8 B4 (hard, informativeness): at least one station passes the gate in the fixed-split
    null run; otherwise V5b is "uninformative", which is not a pass."""
    fixed = _null_runs(results)
    fixed = fixed[(fixed["window"] == "fixed") & (fixed["status"] == "ok")]
    stations = list(fixed.loc[fixed["gate_passed"].eq(True), "station"].unique())
    return {
        "pass": bool(stations),
        "outcome": "informative" if stations else "uninformative",
        "stations_passing_gate": stations,
    }


def _b5_split(runs: pd.DataFrame) -> dict:
    if runs.empty:
        return {"pass": None, "n_runs": 0, "mean_abs_savings_pct": None}
    mean_abs = {m: float(runs[_SAVINGS_COLUMNS[m]].abs().mean()) for m in METHODS}
    ours = mean_abs["saveproof"]
    return {
        "pass": bool(ours < mean_abs["bee_sec"] and ours < mean_abs["naive"]),
        "n_runs": len(runs),
        "mean_abs_savings_pct": mean_abs,
    }


def b5(results: pd.DataFrame) -> dict:
    """§8 B5 (headline comparison): over null runs whose gate passed, SaveProof's mean
    |savings_pct| is strictly lower than both SEC-ratio's and naive's, in the fixed split and
    separately across the rolling windows. pass is False if either split fails, else None if
    either has no such runs, else True."""
    runs = _null_runs(results)
    runs = runs[(runs["status"] == "ok") & runs["gate_passed"].eq(True)]
    splits = {
        "fixed": _b5_split(runs[runs["window"] == "fixed"]),
        "rolling": _b5_split(runs[runs["window"].str.startswith("rolling_")]),
    }
    outcomes = [split["pass"] for split in splits.values()]
    overall = False if False in outcomes else (None if None in outcomes else True)
    return {**splits, "pass": overall}


def _optional_bool(value) -> bool | None:
    return None if pd.isna(value) else bool(value)


def _station_stability(null_runs: pd.DataFrame) -> dict:
    fixed = null_runs.loc[null_runs["window"] == "fixed", "stability_flagged"]
    rolling = null_runs.loc[null_runs["window"].str.startswith("rolling_"), "stability_flagged"]
    return {
        "fixed_flagged": _optional_bool(fixed.iloc[0]) if len(fixed) else None,
        "rolling_flagged": int(rolling.eq(True).sum()),
        "rolling_windows": len(rolling),
    }


def _verified_counts(table: pd.DataFrame, method: str) -> dict:
    counts = _count_verified(table, method)
    del counts["pass"]
    return counts


def _verified_rate(table: pd.DataFrame, method: str) -> dict:
    counts = _verified_counts(table, method)
    n_scored = counts["n_scored"]
    return {
        "verified_rate": counts["n_verified"] / n_scored if n_scored else None,
        "n_scored": n_scored,
        "n_insufficient": counts["n_insufficient"],
    }


def b6(results: pd.DataFrame) -> dict:
    """§8 B6 (reported): comparator false "verified" counts at B6_LOW_SAVING_PCTS; verified rate
    of every method per station at B6_POWER_SAVING_PCTS (None where nothing scored); per station
    the share of scored runs passing the gate and the §10 stability flag of its windows."""
    pct = results["injected_saving_pct"]
    false_verified = {
        method: {str(s): _verified_counts(results[pct == s], method) for s in B6_LOW_SAVING_PCTS}
        for method in ("bee_sec", "naive")
    }
    verified_rate, gate_pass_share, stability = {}, {}, {}
    for station, runs in results.groupby("station", sort=False):
        runs_pct = runs["injected_saving_pct"]
        verified_rate[station] = {
            method: {
                str(s): _verified_rate(runs[runs_pct == s], method) for s in B6_POWER_SAVING_PCTS
            }
            for method in METHODS
        }
        scored, n_insufficient = _scored(runs)
        gate_pass_share[station] = {
            "share": float(scored["gate_passed"].eq(True).mean()) if len(scored) else None,
            "n_scored": len(scored),
            "n_insufficient": n_insufficient,
        }
        stability[station] = _station_stability(_null_runs(runs))
    return {
        "false_verified": false_verified,
        "verified_rate": verified_rate,
        "gate_pass_share": gate_pass_share,
        "stability": stability,
    }
