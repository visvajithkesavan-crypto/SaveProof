"""Score contract-table data quality and report problems without changing data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from saveproof.contract import REQUIRED

ZERO_PRODUCTION_RUN_MIN_PERIODS = 3
KWH_OUTLIER_ROBUST_Z_THRESHOLD = 5.0
KWH_PER_PRODUCTION_SHIFT_THRESHOLD = 0.10
RATIO_CHECK_MIN_ROWS = 3


@dataclass(frozen=True)
class QualityFlag:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class QualityReport:
    total: float
    completeness: float
    continuity: float
    plausibility: float
    consistency: float
    flags: tuple[QualityFlag, ...]


def _date_label(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _gap_flags(
    period_start: pd.Series,
    period_end: pd.Series,
    usable: pd.Series,
    flags: list[QualityFlag],
) -> float:
    first_day = period_start.min().normalize()
    last_day = period_end.max().normalize()
    calendar = pd.date_range(first_day, last_day, freq="D")
    usable_days: set[pd.Timestamp] = set()

    for start, end, is_usable in zip(period_start, period_end, usable, strict=True):
        if is_usable and start <= end:
            usable_days.update(pd.date_range(start.normalize(), end.normalize(), freq="D"))

    missing_days = [day for day in calendar if day not in usable_days]
    if not calendar.empty:
        gap_start = gap_end = None
        for day in [*missing_days, None]:
            if gap_start is not None and (day is None or day - gap_end != pd.Timedelta(days=1)):
                flags.append(
                    QualityFlag(
                        "gap",
                        "problem",
                        f"No usable data from {_date_label(gap_start)} to {_date_label(gap_end)}.",
                    )
                )
                gap_start = None
            if day is not None:
                if gap_start is None:
                    gap_start = day
                gap_end = day

    return 100.0 * len(usable_days.intersection(calendar)) / len(calendar) if len(calendar) else 0.0


def _zero_production_runs(
    ordered: pd.DataFrame,
    kwh: pd.Series,
    production: pd.Series,
    operating_days: pd.Series,
    flags: list[QualityFlag],
) -> pd.Series:
    suspect = pd.Series(False, index=ordered.index)
    run: list[int] = []

    def report_run() -> None:
        if len(run) >= ZERO_PRODUCTION_RUN_MIN_PERIODS:
            first = ordered.loc[run[0], "period_start"]
            last = ordered.loc[run[-1], "period_end"]
            flags.append(
                QualityFlag(
                    "suspect_zero_production",
                    "warning",
                    f"Production is zero from {_date_label(first)} to {_date_label(last)}; "
                    "data left unchanged.",
                )
            )
            suspect.loc[run] = True

    previous_end = None
    for row_index in ordered.index:
        start = ordered.loc[row_index, "period_start"]
        end = ordered.loc[row_index, "period_end"]
        if previous_end is not None and start > previous_end + pd.Timedelta(days=1):
            report_run()
            run = []

        if (
            production.loc[row_index] == 0
            and kwh.loc[row_index] > 0
            and operating_days.loc[row_index] > 0
        ):
            run.append(row_index)
        else:
            report_run()
            run = []
        previous_end = end

    report_run()
    return suspect


def score(
    df: pd.DataFrame, baseline_end: pd.Timestamp | None = None
) -> QualityReport:
    """Score contract rows on a 0-100 scale and return flags without editing ``df``."""
    if df.empty:
        raise ValueError("no rows")
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")

    flags: list[QualityFlag] = []
    period_start = pd.to_datetime(df["period_start"])
    period_end = pd.to_datetime(df["period_end"])
    kwh = pd.to_numeric(df["kwh"], errors="coerce")
    production = pd.to_numeric(df["production"], errors="coerce")
    operating_days = pd.to_numeric(df["operating_days"], errors="coerce")
    coverage_pct = pd.to_numeric(df["coverage_pct"], errors="coerce")

    missing_kwh = kwh.isna()
    for row_index in df.index[missing_kwh]:
        flags.append(
            QualityFlag(
                "missing_kwh",
                "problem",
                f"kwh is missing for {_date_label(period_start.loc[row_index])}.",
            )
        )
    for row_index in df.index[coverage_pct.lt(100).fillna(False)]:
        flags.append(
            QualityFlag(
                "partial_coverage",
                "warning",
                f"Coverage is {coverage_pct.loc[row_index]:g}% for "
                f"{_date_label(period_start.loc[row_index])}.",
            )
        )

    completeness_rows = coverage_pct.fillna(0).clip(0, 100).where(~missing_kwh, 0)
    completeness = float(completeness_rows.mean())
    usable = (~missing_kwh) & coverage_pct.gt(0).fillna(False)
    continuity = _gap_flags(period_start, period_end, usable, flags)

    negative_kwh = kwh.lt(0).fillna(False)
    negative_production = production.lt(0).fillna(False)
    for code, mask, column in (
        ("negative_kwh", negative_kwh, "kwh"),
        ("negative_production", negative_production, "production"),
    ):
        for row_index in df.index[mask]:
            flags.append(
                QualityFlag(
                    code,
                    "problem",
                    f"{column} is negative for {_date_label(period_start.loc[row_index])}.",
                )
            )

    numeric = {
        "kwh": kwh,
        "production": production,
        "operating_days": operating_days,
        "coverage_pct": coverage_pct,
    }
    infinite = pd.Series(False, index=df.index)
    for values in numeric.values():
        infinite |= pd.Series(np.isinf(values), index=df.index)
    for row_index in df.index[infinite]:
        columns = [name for name, values in numeric.items() if np.isinf(values.loc[row_index])]
        flags.append(
            QualityFlag(
                "infinite_value",
                "problem",
                f"Infinite value in {', '.join(columns)} for "
                f"{_date_label(period_start.loc[row_index])}.",
            )
        )

    finite_kwh = kwh[np.isfinite(kwh)]
    if not finite_kwh.empty:
        median = float(finite_kwh.median())
        mad = float((finite_kwh - median).abs().median())
        if mad == 0:
            outlier_mask = np.isfinite(kwh) & kwh.ne(median)
        else:
            robust_z = 0.6745 * (kwh - median).abs() / mad
            outlier_mask = np.isfinite(kwh) & robust_z.gt(KWH_OUTLIER_ROBUST_Z_THRESHOLD)
        for row_index in df.index[outlier_mask.fillna(False)]:
            flags.append(
                QualityFlag(
                    "kwh_outlier",
                    "warning",
                    f"kwh is a robust-MAD outlier for "
                    f"{_date_label(period_start.loc[row_index])}.",
                )
            )

    ordered = df.assign(
        period_start=period_start, period_end=period_end
    ).sort_values("period_start", kind="stable")
    ordered_kwh = kwh.loc[ordered.index]
    ordered_production = production.loc[ordered.index]
    ordered_operating_days = operating_days.loc[ordered.index]
    zero_run = _zero_production_runs(
        ordered,
        ordered_kwh,
        ordered_production,
        ordered_operating_days,
        flags,
    ).reindex(df.index, fill_value=False)
    plausibility_rows = negative_kwh | negative_production | infinite | zero_run
    plausibility = float(100.0 * (1.0 - plausibility_rows.mean()))

    consistency = 100.0
    previous_end = None
    for row_index in ordered.index:
        start = ordered.loc[row_index, "period_start"]
        end = ordered.loc[row_index, "period_end"]
        if previous_end is not None and start <= previous_end:
            flags.append(
                QualityFlag(
                    "overlap",
                    "problem",
                    f"Period starting {_date_label(start)} overlaps the previous period.",
                )
            )
        period_days = (end - start).days + 1
        if operating_days.loc[row_index] > period_days:
            flags.append(
                QualityFlag(
                    "operating_days",
                    "problem",
                    f"operating_days exceeds period length for {_date_label(start)}.",
                )
            )
        previous_end = end

    if baseline_end is not None:
        boundary = pd.Timestamp(baseline_end).normalize()
        no_event = df["event"].isna()
        valid_ratio = (
            no_event
            & production.gt(0)
            & np.isfinite(production)
            & np.isfinite(kwh)
        )
        before = valid_ratio & period_end.le(boundary)
        after = valid_ratio & period_start.gt(boundary)
        before_count = int(before.sum())
        after_count = int(after.sum())
        if min(before_count, after_count) < RATIO_CHECK_MIN_ROWS:
            flags.append(
                QualityFlag(
                    "ratio_check_skipped",
                    "warning",
                    f"kWh-per-production shift check needs at least {RATIO_CHECK_MIN_ROWS} "
                    "eligible rows on each side of baseline_end.",
                )
            )
        else:
            before_median = float((kwh[before] / production[before]).median())
            after_median = float((kwh[after] / production[after]).median())
            if before_median != 0:
                relative_change = abs(after_median - before_median) / abs(before_median)
                if relative_change >= KWH_PER_PRODUCTION_SHIFT_THRESHOLD:
                    message = (
                        f"Median kWh per production changed by {relative_change:.1%} "
                        "across baseline_end."
                    )
                    if df["production_unit"].astype(str).str.startswith("proxy:").any():
                        message += " A proxy change may reflect a power-factor change."
                    flags.append(
                        QualityFlag("kwh_per_production_shift", "problem", message)
                    )
                    consistency = max(0.0, consistency - 50.0)

    total = min(completeness, continuity, plausibility, consistency)
    return QualityReport(
        total=total,
        completeness=completeness,
        continuity=continuity,
        plausibility=plausibility,
        consistency=consistency,
        flags=tuple(flags),
    )
