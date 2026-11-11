"""Null test (docs/validation_plan.md §4): real data, nothing changed, so no window may verify.

Runs every spec (§3) on the fixed split and every rolling window and writes the results table,
the criteria summary and one chart to results/. No randomness, no arguments. From the repo root:
    .\\.venv\\Scripts\\python.exe -m experiments.null_test
"""

import pandas as pd
from matplotlib.figure import Figure

from experiments.common import (
    all_windows,
    current_git_hash,
    load_datasets,
    provenance,
    status_counts,
    window_columns,
    write_outputs,
)
from saveproof.config import ADEETIE_THRESHOLD, CONFIDENCE_REPORT
from saveproof.validation import SPECS, n1, n2, run_window

LABEL = "real"


def run(windows: dict) -> pd.DataFrame:
    """One row per spec and window."""
    rows = []
    for spec, (dataset, formula) in SPECS.items():
        for window in windows[dataset]:
            result = run_window(window.baseline, window.reporting, formula)
            rows.append({**window_columns(spec, window), **result})
    return pd.DataFrame(rows).assign(label=LABEL)


def plot(table: pd.DataFrame) -> Figure:
    """Savings % with its band per window (x = reporting start), one panel per spec."""
    fig = Figure(figsize=(13, 8), layout="constrained")
    axes = fig.subplots(2, 2)
    for ax, (spec, (_, formula)) in zip(axes.flat, SPECS.items(), strict=True):
        rows = table[(table["spec"] == spec) & (table["status"] == "ok")]
        savings = rows["savings_pct"].to_numpy(dtype=float)
        low = rows["band_low_pct"].to_numpy(dtype=float)
        high = rows["band_high_pct"].to_numpy(dtype=float)
        x = pd.to_datetime(rows["reporting_start"]).to_numpy()
        fixed = (rows["window"] == "fixed").to_numpy()
        for mask, style, name in ((~fixed, "o", "rolling"), (fixed, "s", "fixed split")):
            ax.errorbar(
                x[mask],
                savings[mask],
                yerr=[(savings - low)[mask], (high - savings)[mask]],
                fmt=style,
                capsize=3,
                label=name,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(100 * ADEETIE_THRESHOLD, color="red", linestyle="--", label="10% (ADEETIE)")
        n_insufficient = int((table["spec"] == spec).sum()) - len(rows)
        ax.set_title(f"{spec}: {formula}\n{len(rows)} windows, {n_insufficient} insufficient data")
        ax.set_xlabel("reporting window start")
        ax.set_ylabel(f"savings % ({CONFIDENCE_REPORT:.0%} band)")
        ax.legend(fontsize=8)
    fig.suptitle("Null test: real data, nothing changed (label: real)")
    return fig


def main() -> None:
    git_hash = current_git_hash()
    frames = load_datasets()
    table = run(all_windows(frames))
    summary = {
        **provenance("null_test", LABEL, git_hash, frames),
        "counts": status_counts(table),
        "criteria": {
            "N1": {
                "plan_wording": "zero verified verdicts in every null window for every spec",
                "hard": True,
                **n1(table),
            },
            "N2": {
                "plan_wording": "the share of windows whose band contains 0, per spec; an adequate "
                "spec (UCI-B or IR-B) below 50% goes on the results slide as a limitation",
                "hard": False,
                "per_spec": n2(table),
            },
            "N3": {
                "plan_wording": "CV(RMSE), FSU at 10%, R2, NMBE, lag-1 rho and n' for every window",
                "hard": False,
                "reported_in": "CSV columns cvrmse, fsu, r2, nmbe, rho, n_eff",
            },
        },
    }
    for path in write_outputs("null_test", table, summary, plot(table)):
        print(path)
    print(f"N1 pass: {summary['criteria']['N1']['pass']}")


if __name__ == "__main__":
    main()
