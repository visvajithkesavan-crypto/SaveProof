"""Comparators SaveProof is measured against: a naive bill comparison and the BEE SEC-ratio.

Neither has an uncertainty band, so the verdict comes from the point estimate alone
(docs/validation_plan.md §2). Rows are dropped with the same contract.validate and
baseline.usable_rows as verify_savings. savings_pct is PERCENT (0-100).
"""

from dataclasses import asdict, dataclass

import pandas as pd

from saveproof.config import ADEETIE_THRESHOLD
from saveproof.contract import validate
from saveproof.engine.baseline import add_calendar_terms, usable_rows


@dataclass(frozen=True)
class ComparatorResult:
    """Point-estimate savings over the usable reporting periods (kWh; percent of expected kWh)."""

    method: str  # "naive" | "bee_sec"
    savings_kwh: float  # expected - actual reporting kWh
    savings_pct: float
    verdict: str  # "verified" if savings_pct >= 100 * ADEETIE_THRESHOLD, else "below"

    def to_dict(self) -> dict:
        """JSON-serialisable view of every field."""
        return asdict(self)


def _usable(df: pd.DataFrame, formula: str, label: str) -> pd.DataFrame:
    """Validated rows of df (with period_days) that usable_rows keeps; raises if none are left."""
    rows = add_calendar_terms(validate(df))
    rows = rows[usable_rows(rows, formula)]
    if rows.empty:
        raise ValueError(f"{label} frame has no usable rows (all have an event or low coverage)")
    return rows


def _result(method: str, expected_kwh: float, actual_kwh: float) -> ComparatorResult:
    if expected_kwh <= 0:
        raise ValueError(
            f"{method}: expected reporting kWh is {expected_kwh:.1f}, must be positive"
        )
    savings_kwh = expected_kwh - actual_kwh
    savings_pct = 100 * savings_kwh / expected_kwh
    verdict = "verified" if savings_pct >= 100 * ADEETIE_THRESHOLD else "below"
    return ComparatorResult(method, float(savings_kwh), float(savings_pct), verdict)


def naive_savings(
    baseline_df: pd.DataFrame, reporting_df: pd.DataFrame, formula: str = "kwh ~ 1"
) -> ComparatorResult:
    """Bill comparison normalised for period length, ignoring production: expected reporting kWh =
    baseline kWh per calendar day x reporting calendar days. Contract frames in, kWh out.

    formula only selects rows (usable_rows); pass the SaveProof model's formula so naive drops
    the same rows as verify_savings (e.g. NaN production).
    """
    baseline = _usable(baseline_df, formula, "baseline")
    reporting = _usable(reporting_df, formula, "reporting")
    baseline_kwh_per_day = baseline["kwh"].sum() / baseline["period_days"].sum()
    expected_kwh = baseline_kwh_per_day * reporting["period_days"].sum()
    return _result("naive", expected_kwh, reporting["kwh"].sum())


def bee_sec_savings(baseline_df: pd.DataFrame, reporting_df: pd.DataFrame) -> ComparatorResult:
    """BEE SEC-ratio: savings_kwh = (SEC_base - SEC_report) x reporting production, with
    SEC = total kWh / total production. Assumes all energy scales with output, so fixed loads
    make it report false savings when production rises (mv-engine skill, "Comparators").

    savings_pct = 100 x (1 - SEC_report / SEC_base), computed as savings over SEC_base x
    reporting production (the same quantity). Raises ValueError if production_unit differs
    between the frames or a frame's total usable production is 0.
    """
    baseline = _usable(baseline_df, "kwh ~ production", "baseline")
    reporting = _usable(reporting_df, "kwh ~ production", "reporting")
    base_unit = baseline["production_unit"].iloc[0]
    report_unit = reporting["production_unit"].iloc[0]
    if base_unit != report_unit:
        raise ValueError(
            f"column 'production_unit': baseline is {base_unit!r}, reporting is {report_unit!r}"
        )
    for label, rows in (("baseline", baseline), ("reporting", reporting)):
        if rows["production"].sum() == 0:
            raise ValueError(f"column 'production': {label} total over usable rows is 0")
    sec_base_kwh_per_unit = baseline["kwh"].sum() / baseline["production"].sum()
    expected_kwh = sec_base_kwh_per_unit * reporting["production"].sum()
    return _result("bee_sec", expected_kwh, reporting["kwh"].sum())
