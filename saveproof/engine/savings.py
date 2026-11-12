"""Verified savings: adjusted baseline minus actual kWh over the reporting period, an uncertainty
band, and the ADEETIE verdict. The verdict rule is docs/validation_plan.md §2 plus the proxy gate
of docs/validation_addendum_v5b.md §9 (SaveProof's own proposal; BEE does not define how to treat
uncertainty).

Reporting-period frame in, SavingsResult out. All *_pct fields are PERCENT (0-100), so compare
them with 100 * ADEETIE_THRESHOLD. Dates are plant-local (contract).
"""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from saveproof.config import (
    ADEETIE_THRESHOLD,
    CONFIDENCE_ASHRAE,
    CONFIDENCE_REPORT,
    CVRMSE_GUIDE,
    FSU_MAX_ASHRAE,
    MAX_EXCLUDED_SHARE,
    MAX_EXTRAPOLATED_SHARE,
    PROD_RANGE_HIGH_FACTOR,
    PROD_RANGE_LOW_FACTOR,
)
from saveproof.contract import validate
from saveproof.engine import stats
from saveproof.engine.baseline import (
    BaselineModel,
    InsufficientDataError,
    add_calendar_terms,
    is_monthly,
    predict,
    usable_rows,
)


@dataclass(frozen=True)
class SavingsResult:
    """Savings over the usable reporting periods. kWh for energy, percent of the adjusted
    baseline for *_pct, fractions for fsu and extrapolated_share."""

    adjusted_baseline_kwh: float  # model prediction at the reporting period's own production
    actual_kwh: float
    savings_kwh: float  # adjusted_baseline_kwh - actual_kwh
    savings_pct: float
    band_low_pct: float  # two-sided band at `confidence`
    band_high_pct: float
    confidence: float
    fsu: float  # FSU at CONFIDENCE_ASHRAE, evaluated at a 10% saving (see verify_savings)
    n: int  # baseline periods behind the model
    m: int  # usable reporting periods
    m_excluded: int  # reporting periods dropped by usable_rows
    extrapolated_share: float  # share of usable rows with production outside the baseline range
    ashrae_ok: bool  # adequacy gate: CV(RMSE) and FSU
    verdict: str  # "verified" | "likely_not_proven" | "below"
    gate_failures: tuple[str, ...]  # human-readable reasons a gate failed; empty if both passed

    def to_dict(self) -> dict:
        """JSON-serialisable view of every field (gate_failures becomes a list)."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["gate_failures"] = list(self.gate_failures)
        return out


PROXY_GATE_FAILURE = "production variable is not measured output (proxy or absent)"


def production_is_measured(model: BaselineModel) -> bool:
    """Proxy gate (addendum §9): the formula has a production term and production_unit does not
    start with "proxy:" (loaders label electrical, estimated, calendar and modelled values so)."""
    has_production_term = not np.isnan(model.production_min)
    return has_production_term and not model.production_unit.startswith("proxy:")


def decide_verdict(
    band_low_pct: float,
    band_high_pct: float,
    ashrae_ok: bool,
    extrapolated_share: float,
    production_measured: bool,
) -> str:
    """Verdict per docs/validation_plan.md §2 steps 3-4 and the proxy gate of addendum §9. Band in
    percent of the adjusted baseline.

    Any gate failing gives "likely_not_proven": a bad model can never verify or rule out 10%.
    Otherwise "verified" if the whole band is at or above 10%, "below" if it is entirely below.
    """
    gates_ok = ashrae_ok and extrapolated_share <= MAX_EXTRAPOLATED_SHARE and production_measured
    if not gates_ok:
        return "likely_not_proven"
    threshold_pct = 100 * ADEETIE_THRESHOLD
    if band_low_pct >= threshold_pct:
        return "verified"
    if band_high_pct < threshold_pct:
        return "below"
    return "likely_not_proven"


def _extrapolated_share(model: BaselineModel, rows: pd.DataFrame) -> float:
    """Share of rows with production outside the baseline range widened by the PROD_RANGE_*
    factors; 0.0 when the formula has no production term."""
    if np.isnan(model.production_min):
        return 0.0
    low = PROD_RANGE_LOW_FACTOR * model.production_min
    high = PROD_RANGE_HIGH_FACTOR * model.production_max
    return float(((rows["production"] < low) | (rows["production"] > high)).mean())


def _ashrae_failures(cvrmse: float, fsu: float) -> list[str]:
    failures = []
    if cvrmse > CVRMSE_GUIDE:
        failures.append(f"CV(RMSE) {cvrmse:.2f} > {CVRMSE_GUIDE:.2f}")
    if fsu > FSU_MAX_ASHRAE:
        failures.append(f"FSU at 10% saving {fsu:.2f} > {FSU_MAX_ASHRAE:.2f}")
    return failures


def _granularity(monthly: bool) -> str:
    return "monthly (multi-day periods)" if monthly else "daily"


def _usable_reporting_rows(model: BaselineModel, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Validated reporting rows that usable_rows keeps, and how many it dropped. Raises as
    documented in verify_savings."""
    reporting = add_calendar_terms(validate(df))
    if is_monthly(reporting) != model.monthly:
        raise ValueError(
            f"granularity mismatch: model is {_granularity(model.monthly)}, "
            f"reporting frame is {_granularity(not model.monthly)}"
        )
    usable = usable_rows(reporting, model.formula)
    m_excluded = int((~usable).sum())
    if m_excluded / len(reporting) > MAX_EXCLUDED_SHARE:
        raise InsufficientDataError(
            f"{m_excluded} of {len(reporting)} reporting periods excluded "
            f"({m_excluded / len(reporting):.0%}), more than the allowed {MAX_EXCLUDED_SHARE:.0%}"
        )
    return reporting[usable], m_excluded


def verify_savings(
    model: BaselineModel, df: pd.DataFrame, confidence: float = CONFIDENCE_REPORT
) -> SavingsResult:
    """Savings of the reporting-period contract frame `df` against `model` (kWh; band in percent).

    Raises ValueError for contract violations or a daily/monthly mismatch with the model, and
    InsufficientDataError when more than MAX_EXCLUDED_SHARE of the periods are unusable.
    """
    rows, m_excluded = _usable_reporting_rows(model, df)
    m = len(rows)

    adjusted_baseline_kwh = float(predict(model, rows).sum())
    if adjusted_baseline_kwh <= 0:
        raise ValueError(f"adjusted baseline is {adjusted_baseline_kwh:.1f} kWh, must be positive")
    actual_kwh = float(rows["kwh"].sum())
    savings_kwh = adjusted_baseline_kwh - actual_kwh

    def half_width_kwh(level: float) -> float:
        return stats.savings_uncertainty_kwh(
            model.cvrmse, model.n, model.p, model.rho, m, adjusted_baseline_kwh, level,
            model.monthly,
        )  # fmt: skip

    half_width_pct = 100 * half_width_kwh(confidence) / adjusted_baseline_kwh
    savings_pct = 100 * savings_kwh / adjusted_baseline_kwh
    # FSU at the ASHRAE confidence, evaluated at a 10% saving instead of the measured one: the
    # absolute uncertainty does not depend on the saving, so this asks "could this model verify
    # 10% at all?" independent of what was measured (validation_plan §2 step 1).
    fsu = half_width_kwh(CONFIDENCE_ASHRAE) / (ADEETIE_THRESHOLD * adjusted_baseline_kwh)

    ashrae_failures = _ashrae_failures(model.cvrmse, fsu)
    extrapolated_share = _extrapolated_share(model, rows)
    extrapolation_failures = (
        [f"{extrapolated_share:.0%} of reporting periods outside the baseline production range"]
        if extrapolated_share > MAX_EXTRAPOLATED_SHARE
        else []
    )
    production_measured = production_is_measured(model)
    proxy_failures = [] if production_measured else [PROXY_GATE_FAILURE]
    band_low_pct = savings_pct - half_width_pct
    band_high_pct = savings_pct + half_width_pct
    ashrae_ok = not ashrae_failures
    return SavingsResult(
        adjusted_baseline_kwh=adjusted_baseline_kwh,
        actual_kwh=actual_kwh,
        savings_kwh=savings_kwh,
        savings_pct=savings_pct,
        band_low_pct=band_low_pct,
        band_high_pct=band_high_pct,
        confidence=confidence,
        fsu=fsu,
        n=model.n,
        m=m,
        m_excluded=m_excluded,
        extrapolated_share=extrapolated_share,
        ashrae_ok=ashrae_ok,
        verdict=decide_verdict(
            band_low_pct, band_high_pct, ashrae_ok, extrapolated_share, production_measured
        ),
        gate_failures=(*ashrae_failures, *extrapolation_failures, *proxy_failures),
    )
