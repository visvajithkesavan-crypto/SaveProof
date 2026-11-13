"""Baseline stability: is there a material, significant time trend in the baseline residuals?

Method and thresholds: docs/validation_addendum_v5b.md §10. A reported flag, not a verdict gate.
Daily baselines only; t = calendar days since the first usable baseline date (plant-local).
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import statsmodels.api as sm

from saveproof.config import STABILITY_DRIFT_SHARE_MIN, STABILITY_P_MAX
from saveproof.engine.baseline import BaselineModel


@dataclass(frozen=True)
class StabilityResult:
    """Trend of baseline residuals: slope in kWh per day, p two-sided, drift_share a fraction of
    mean baseline kWh."""

    slope_kwh_per_day: float
    p_value: float
    drift_share: float  # |slope| x (n - 1) / mean baseline kWh
    flagged: bool  # "baseline unstable": p < STABILITY_P_MAX and drift_share >= ..._MIN

    def to_dict(self) -> dict:
        """JSON-serialisable view of every field."""
        return asdict(self)


def newey_west_lag(n: int) -> int:
    """Lag L = floor(4 * (n / 100) ** (2 / 9)) for n residuals (addendum §10)."""
    return math.floor(4 * (n / 100) ** (2 / 9))


def is_unstable(p_value: float, drift_share: float) -> bool:
    """Both conditions of addendum §10: significance (p < 0.05) and materiality (D >= 0.05)."""
    return p_value < STABILITY_P_MAX and drift_share >= STABILITY_DRIFT_SHARE_MIN


def baseline_stability(model: BaselineModel) -> StabilityResult:
    """Regress the model's residuals, in time order, on days since baseline start; test the slope
    with Newey-West standard errors (Bartlett kernel, lag newey_west_lag(n), no small-sample
    correction, normal reference distribution, two-sided). Raises ValueError for monthly models.
    """
    if model.monthly:
        raise ValueError(
            "baseline_stability is defined for daily baselines: D = |slope| x (n - 1) assumes "
            "one period per day (addendum §10)"
        )
    frame = model.result.model.data.frame.assign(resid_kwh=model.result.resid)
    frame = frame.sort_values("period_start")
    days = (frame["period_start"] - frame["period_start"].iloc[0]).dt.days.to_numpy(dtype=float)
    n = len(frame)
    trend = sm.OLS(frame["resid_kwh"].to_numpy(), sm.add_constant(days)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": newey_west_lag(n), "kernel": "bartlett", "use_correction": False},
        use_t=False,
    )
    slope_kwh_per_day = float(trend.params[1])
    p_value = float(trend.pvalues[1])
    drift_share = abs(slope_kwh_per_day) * (n - 1) / float(np.mean(frame["kwh"]))
    return StabilityResult(
        slope_kwh_per_day=slope_kwh_per_day,
        p_value=p_value,
        drift_share=drift_share,
        flagged=is_unstable(p_value, drift_share),
    )
