"""Validation experiments as pure functions: model specs, null-test windows, injected savings and
the pre-registered pass criteria. The authority is docs/validation_plan.md §3 (specs), §4 (null
test), §5 (injected saving) and §8 (I2 amendment); every constant here cites it.

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
from saveproof.engine.savings import SavingsResult, verify_savings

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
