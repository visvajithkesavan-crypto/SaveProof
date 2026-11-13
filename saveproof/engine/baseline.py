"""Baseline model: OLS of kWh on production / calendar terms, fitted on baseline usable rows only.

Statistics (CV(RMSE), NMBE, rho, n') come from engine/stats.py. Timestamps are plant-local
dates (contract); `period_start` marks the first day of each period.
"""

import re
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from saveproof.config import MAX_EXCLUDED_SHARE, MIN_BASELINE_DOF, MIN_COVERAGE_PCT
from saveproof.contract import validate
from saveproof.engine import stats


class InsufficientDataError(ValueError):
    """The baseline has too few usable rows to fit (too many excluded, or n - p too small)."""


@dataclass(frozen=True)
class BaselineModel:
    """Fitted baseline. kWh units for params (kwh per unit of each term); fractions for the rest."""

    formula: str
    params: dict[str, float]
    n: int  # usable baseline periods used in the fit
    p: int  # parameters incl. intercept
    r2: float
    cvrmse: float  # fraction of mean kWh
    nmbe: float  # fraction of mean kWh
    rho: float  # lag-1 autocorrelation of residuals in period order
    n_eff: float  # n' from max(rho, 0)
    production_min: float  # over usable rows, in the production_unit; NaN if no production term
    production_max: float
    production_unit: str  # of the baseline frame, e.g. "t" or "proxy:kvarh_lagging"
    monthly: bool  # True if the baseline periods are longer than one day
    n_excluded: int  # baseline periods dropped by usable_rows
    result: object = field(repr=False, compare=False)  # statsmodels results, used by predict

    def to_dict(self) -> dict:
        """JSON-serialisable view of every field except the statsmodels result."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "result"}


def add_calendar_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with two derived columns (neither is a contract column):
    day_type = "weekend" if period_start is Sat/Sun, else "weekday" (docs/validation_plan.md §3;
    meaningful for daily periods only), and period_days = calendar days in the period, end
    inclusive (1 for a daily row)."""
    out = df.copy()
    is_weekend = out["period_start"].dt.dayofweek >= 5
    out["day_type"] = np.where(is_weekend, "weekend", "weekday")
    out["period_days"] = (out["period_end"] - out["period_start"]).dt.days + 1
    return out


def _regressor_columns(df: pd.DataFrame, formula: str) -> list[str]:
    """Columns of df named on the right-hand side of formula (the response is not a regressor)."""
    if "~" not in formula:
        raise ValueError(f"formula must look like 'kwh ~ ...', got {formula!r}")
    names = set(re.findall(r"[A-Za-z_]\w*", formula.split("~", 1)[1]))
    return [c for c in df.columns if c in names]


def usable_rows(df: pd.DataFrame, formula: str) -> pd.Series:
    """Boolean Series aligned to df.index: False where event is set, coverage_pct is below
    MIN_COVERAGE_PCT, or a regressor in formula is NaN (e.g. production)."""
    regressors = _regressor_columns(df, formula)
    has_event = df["event"].notna()
    low_coverage = df["coverage_pct"] < MIN_COVERAGE_PCT
    missing_regressor = df[regressors].isna().any(axis=1)
    return ~(has_event | low_coverage | missing_regressor)


def is_monthly(df: pd.DataFrame) -> bool:
    days_in_period = df["period_days"]
    if (days_in_period == 1).all():
        return False
    if (days_in_period > 1).all():
        return True
    raise ValueError(
        f"mixed daily and longer periods ({int((days_in_period == 1).sum())} one-day rows, "
        f"{int((days_in_period > 1).sum())} longer): fit one granularity at a time"
    )


def _check_design_rank(model, formula: str) -> None:
    """Raise if the design matrix is singular; statsmodels would otherwise silently use pinv."""
    design = pd.DataFrame(model.exog, columns=model.exog_names)
    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank == design.shape[1]:
        return
    constant = [c for c in design.columns if c != "Intercept" and np.ptp(design[c]) == 0]
    if constant:
        reason = (
            f"term(s) {constant} are constant in the usable baseline rows (e.g. operating_days is "
            "always 1 in daily data; use C(day_type))"
        )
    else:
        reason = "some terms are collinear with each other or with the intercept"
    raise ValueError(
        f"formula {formula!r} gives a singular design (rank {rank} of {design.shape[1]} columns "
        f"{list(design.columns)}): {reason}"
    )


def fit_baseline(
    df: pd.DataFrame, formula: str = "kwh ~ production + operating_days"
) -> BaselineModel:
    """Fit OLS `formula` on the usable rows of a contract table (baseline period only).

    Input: contract frame (kWh per period). Output: BaselineModel. Raises ValueError for contract
    violations, mixed granularity and singular designs; InsufficientDataError when too many rows
    are excluded or n - p < MIN_BASELINE_DOF.
    """
    df = add_calendar_terms(validate(df))  # validate returns rows in period order
    monthly = is_monthly(df)
    if monthly and "day_type" in _regressor_columns(df, formula):
        raise ValueError(
            f"formula {formula!r} uses day_type, which is defined by a single date and is "
            "meaningless for multi-day periods; use operating_days/period_days"
        )

    usable = usable_rows(df, formula)
    n_excluded = int((~usable).sum())
    excluded_share = n_excluded / len(df)
    if excluded_share > MAX_EXCLUDED_SHARE:
        raise InsufficientDataError(
            f"{n_excluded} of {len(df)} baseline periods excluded ({excluded_share:.0%}), "
            f"more than the allowed {MAX_EXCLUDED_SHARE:.0%}"
        )

    rows = df[usable]
    ols = smf.ols(formula, data=rows)
    _check_design_rank(ols, formula)
    n, p = ols.exog.shape
    if n - p < MIN_BASELINE_DOF:
        raise InsufficientDataError(
            f"n - p = {n} - {p} = {n - p} usable degrees of freedom, "
            f"need at least {MIN_BASELINE_DOF} ({n_excluded} of {len(df)} periods excluded)"
        )
    result = ols.fit()

    y_kwh = rows["kwh"].to_numpy()
    resid_kwh = result.resid.to_numpy()
    rho = stats.lag1_autocorr(resid_kwh)
    has_production = "production" in _regressor_columns(rows, formula)
    return BaselineModel(
        formula=formula,
        params={name: float(value) for name, value in result.params.items()},
        n=int(n),
        p=int(p),
        r2=float(result.rsquared),
        cvrmse=stats.cvrmse(y_kwh, y_kwh - resid_kwh, p),
        nmbe=stats.nmbe(y_kwh, y_kwh - resid_kwh, p),
        rho=rho,
        n_eff=stats.n_effective(n, max(rho, 0.0)),
        production_min=float(rows["production"].min()) if has_production else float("nan"),
        production_max=float(rows["production"].max()) if has_production else float("nan"),
        production_unit=str(rows["production_unit"].iloc[0]),
        monthly=monthly,
        n_excluded=n_excluded,
        result=result,
    )


def predict(model: BaselineModel, df: pd.DataFrame) -> pd.Series:
    """Baseline kWh per period of df, named "kwh_pred" and aligned to df.index.

    Needs the formula's regressors in df (day_type is added here). Raises ValueError naming the
    column if a regressor is NaN, or the level if day_type has one the baseline never saw.
    """
    df = add_calendar_terms(df)
    regressors = _regressor_columns(df, model.formula)
    for column in regressors:
        missing = df[column].isna()
        if missing.any():
            raise ValueError(
                f"column '{column}': row {df.index[missing.to_numpy().argmax()]} is NaN"
            )
    if "day_type" in regressors:
        seen = set(model.result.model.data.frame["day_type"])
        unseen = sorted(set(df["day_type"]) - seen)
        if unseen:
            raise ValueError(
                f"day_type level(s) {unseen} not in the baseline (baseline had {sorted(seen)})"
            )
    kwh_pred = np.asarray(model.result.predict(df))
    return pd.Series(kwh_pred, index=df.index, name="kwh_pred")
