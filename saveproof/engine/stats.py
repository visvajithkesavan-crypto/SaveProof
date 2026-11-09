"""ASHRAE Guideline 14 statistics for baseline models. Pure numpy/scipy; formulas as in
.cursor/skills/mv-engine/references/ashrae14-formulas.md.

NOTE (from that reference): verify the FSU expression against the Guideline text before quoting
it in the deck.
"""

import numpy as np
from scipy import stats as scipy_stats

from saveproof.config import (
    FSU_DAILY_ALPHA_COEFFS,
    FSU_DAILY_DAYS_PER_MONTH,
    FSU_MONTHLY_FACTOR,
)


def _float_array(name: str, values: np.ndarray) -> np.ndarray:
    """Return `values` as a 1-D float array; raise naming `name` and the first non-finite row."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    bad_rows = np.flatnonzero(~np.isfinite(arr))
    if bad_rows.size:
        raise ValueError(f"{name} has a non-finite value at row {bad_rows[0]}")
    return arr


def _check_fit_inputs(y: np.ndarray, yhat: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    y_arr = _float_array("y", y)
    yhat_arr = _float_array("yhat", yhat)
    if len(y_arr) != len(yhat_arr):
        raise ValueError(f"yhat has {len(yhat_arr)} rows but y has {len(y_arr)}")
    if len(y_arr) <= p:
        raise ValueError(f"p ({p}) must be smaller than the number of rows n ({len(y_arr)})")
    if y_arr.mean() == 0:
        raise ValueError("y has mean 0, so CV(RMSE) and NMBE are undefined")
    return y_arr, yhat_arr


def cvrmse(y: np.ndarray, yhat: np.ndarray, p: int) -> float:
    """CV(RMSE) = sqrt(sum (y - yhat)^2 / (n - p)) / mean(y). Inputs: measured and predicted kWh
    of the baseline period, p model parameters incl. intercept. Output: fraction (0.2 = 20%)."""
    y_arr, yhat_arr = _check_fit_inputs(y, yhat, p)
    rmse_kwh = np.sqrt(np.sum((y_arr - yhat_arr) ** 2) / (len(y_arr) - p))
    return float(rmse_kwh / y_arr.mean())


def nmbe(y: np.ndarray, yhat: np.ndarray, p: int) -> float:
    """NMBE = sum (y - yhat) / ((n - p) * mean(y)). Inputs as `cvrmse`. Output: fraction;
    positive means the model under-predicts."""
    y_arr, yhat_arr = _check_fit_inputs(y, yhat, p)
    return float(np.sum(y_arr - yhat_arr) / ((len(y_arr) - p) * y_arr.mean()))


def lag1_autocorr(resid: np.ndarray) -> float:
    """Lag-1 autocorrelation rho = corr(e_t, e_{t-1}) of baseline residuals (kWh, in time order).
    Output: dimensionless, in [-1, 1]."""
    e = _float_array("resid", resid)
    if len(e) < 3:
        raise ValueError(f"resid needs at least 3 rows, got {len(e)}")
    current, lagged = e[1:], e[:-1]
    if np.ptp(current) == 0 or np.ptp(lagged) == 0:
        raise ValueError("resid is constant, so its autocorrelation is undefined")
    return float(np.corrcoef(current, lagged)[0, 1])


def n_effective(n: float, rho: float) -> float:
    """Effective independent points n' = n (1 - rho) / (1 + rho). Inputs: n baseline points,
    lag-1 autocorrelation rho in (-1, 1). Output: points (n' < n when rho > 0)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not -1 < rho < 1:
        raise ValueError(f"rho must be strictly between -1 and 1, got {rho}")
    return float(n * (1 - rho) / (1 + rho))


def savings_uncertainty_kwh(
    cv: float,
    n: int,
    p: int,
    rho: float,
    m: int,
    adjusted_baseline_kwh: float,
    confidence: float,
    monthly: bool,
) -> float:
    """Absolute savings uncertainty (half-width of the band) in kWh over the reporting period.

    cv: baseline CV(RMSE) as a fraction; n, p: baseline points and parameters; rho: baseline
    lag-1 autocorrelation; m: reporting-period points; adjusted_baseline_kwh: model prediction
    summed over the reporting period; confidence: two-sided level in (0, 1).
    It takes no savings argument: the absolute form does not depend on how large the saving is
    (use it when savings ~ 0, e.g. the null test). Fractional form: divide by savings kWh.
    """
    for name, value in (("cv", cv), ("adjusted_baseline_kwh", adjusted_baseline_kwh)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite number >= 0, got {value}")
    if n <= p:
        raise ValueError(f"n ({n}) must be greater than p ({p})")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    if not -1 < rho < 1:  # checked before clamping so an invalid rho is never hidden
        raise ValueError(f"rho must be strictly between -1 and 1, got {rho}")
    # Deliberate deviation from nmecr: its rho >= 0.5 switch makes the band jump ~1.8x between
    # rho 0.49 and 0.51, and our UCI residuals sit at rho ~0.47-0.50. Negative rho is clamped to 0
    # (conservative).
    rho_used = max(rho, 0.0)
    n_dash = n_effective(n, rho_used)
    upper_tail = 1 - (1 - confidence) / 2  # two-sided Student-t

    if monthly:
        # ASHRAE G14 form as in the mv-engine reference; Reddy & Claridge 1.26 constant.
        t = scipy_stats.t.ppf(upper_tail, n - p)
        return float(
            FSU_MONTHLY_FACTOR
            * t
            * cv
            * np.sqrt((n / n_dash) * (1 + 2 / n) / m)
            * adjusted_baseline_kwh
        )

    # Daily form, ported from nmecr (kW-Labs, MIT), calculate_savings_and_uncertainty.R:
    # dE = t' * alpha * sqrt(MSE' * (1 + 2/n') * m) with MSE' = SSE / (n' - p) and t' at
    # (n' - p) degrees of freedom. With SSE = cv^2 * mean_kwh^2 * (n - p) and
    # adjusted_baseline_kwh = m * mean_kwh this becomes the expression below.
    dof_dash = n_dash - p
    if dof_dash <= 0:
        raise ValueError(f"rho ({rho}) leaves n' = {n_dash:.2f} <= p ({p}): no degrees of freedom")
    x = m / FSU_DAILY_DAYS_PER_MONTH
    if x > 12:
        raise ValueError(
            f"m ({m}) is {x:.1f} months of daily points: "
            "factor is only used for 1-12 month reporting periods"
        )
    a, b, c = FSU_DAILY_ALPHA_COEFFS
    alpha = a * x**2 + b * x + c
    t_dash = scipy_stats.t.ppf(upper_tail, dof_dash)
    return float(
        alpha
        * t_dash
        * cv
        * np.sqrt((n - p) / dof_dash)
        * np.sqrt((1 + 2 / n_dash) / m)
        * adjusted_baseline_kwh
    )
