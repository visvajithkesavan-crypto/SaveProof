"""Baseline stability: is there a material, significant time trend in the baseline residuals?

Method and thresholds: docs/validation_addendum_v5b.md §10. A reported flag, not a verdict gate.
t = calendar days from the first baseline period start to each period start (plant-local).
Monthly D uses that span; daily D stays the addendum's (n - 1). docs/validation_plan.md §8,
2026-11-14.
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
    drift_share: float  # |slope| x span_days / mean baseline kWh per period
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
    """Regress the model's residuals, in time order, on days since the first baseline period
    start; test the slope with Newey-West standard errors (Bartlett kernel, lag
    newey_west_lag(n), no small-sample correction, normal reference distribution, two-sided).

    D = |slope| x span_days / mean baseline kWh per period. Daily span_days is (n - 1), the
    addendum §10 formula, so a gap does not change a daily result. Monthly span_days is
    (t_last - t_first): monthly periods are not one day apart (validation_plan.md §8, 2026-11-14).
    """
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
    span_days = float(days[-1] - days[0]) if model.monthly else float(n - 1)
    drift_share = abs(slope_kwh_per_day) * span_days / float(np.mean(frame["kwh"]))
    return StabilityResult(
        slope_kwh_per_day=slope_kwh_per_day,
        p_value=p_value,
        drift_share=drift_share,
        flagged=is_unstable(p_value, drift_share),
    )
