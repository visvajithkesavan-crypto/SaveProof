"""V5b (docs/validation_addendum_v5b.md): Laayati mine, real production KPI, null and injection.

Per station (destoning, screening, loading): the fixed split and 12 rolling windows (§6), reporting
kWh cut uniformly by s in {0, 5, 7, 10, 12, 15}% (§7), SaveProof with `kwh ~ production` (§5),
naive and BEE SEC-ratio on the same frames, and the §10 stability flag of each baseline. Writes the
results table, the B1-B6 summary (§8) and one chart to results/. No randomness, no arguments. Run
once, from a clean commit, from the repo root (§12):
    .\\.venv\\Scripts\\python.exe -m experiments.v5b
Reads data/raw/laayati/<Station>_hourly_data.csv (the MDPI supplement files, §3).
"""

import pandas as pd
from matplotlib.figure import Figure

from experiments.common import (
    REPO,
    current_git_hash,
    last_commit,
    provenance,
    write_outputs,
)
from saveproof.config import ADEETIE_THRESHOLD
from saveproof.ingest.laayati import STATIONS, load_laayati
from saveproof.validation import (
    METHODS,
    V5B_FORMULA,
    V5B_SAVING_PCTS,
    V5B_SHAPE,
    VERDICT_COLUMNS,
    Window,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    inject_saving,
    run_v5b_window,
    v5b_windows,
    window_stability,
)

LABEL_NULL = "real"
LABEL_INJECTED = "injected (real noise, synthetic saving)"
ADDENDUM_PATH = "docs/validation_addendum_v5b.md"
LAAYATI_DIR = REPO / "data" / "raw" / "laayati"


def load_stations() -> dict[str, pd.DataFrame]:
    """Daily contract frame per station, keyed by station name."""
    return {
        station: load_laayati(LAAYATI_DIR / f"{station.capitalize()}_hourly_data.csv", station)
        for station in STATIONS
    }


def _window_columns(station: str, dataset: str, window: Window) -> dict:
    """Identifying columns of one results row; dates are plant-local, ends inclusive."""
    return {
        "station": station,
        "dataset": dataset,
        "formula": V5B_FORMULA,
        "window": window.name,
        "baseline_start": window.baseline_start.date().isoformat(),
        "baseline_end": window.baseline_end.date().isoformat(),
        "reporting_start": window.reporting_start.date().isoformat(),
        "reporting_end": window.reporting_end.date().isoformat(),
        "baseline_rows": len(window.baseline),
        "reporting_rows": len(window.reporting),
    }


def run(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per station, window and injected saving: SaveProof, comparators, stability."""
    rows = []
    for station, df in frames.items():
        for window in v5b_windows(df):
            ids = _window_columns(station, df["dataset"].iloc[0], window)
            stability = window_stability(window.baseline, V5B_FORMULA)
            for pct in V5B_SAVING_PCTS:
                reporting = inject_saving(window.reporting, V5B_FORMULA, pct / 100, V5B_SHAPE)
                rows.append(
                    {
                        **ids,
                        "injected_saving_pct": pct,
                        "label": LABEL_NULL if pct == 0 else LABEL_INJECTED,
                        **run_v5b_window(window.baseline, reporting, V5B_FORMULA),
                        **stability,
                    }
                )
    return pd.DataFrame(rows)


def _counts(table: pd.DataFrame) -> dict:
    out = {}
    for station, group in table.groupby("station", sort=False):
        out[station] = {
            "runs": len(group),
            "ok": int((group["status"] == "ok").sum()),
            "insufficient_data": int((group["status"] == "insufficient_data").sum()),
        }
    return out


def summarise(table: pd.DataFrame, frames: dict, git_hash: str) -> dict:
    """Provenance, counts and criteria B1-B6 with the addendum's wording (§8)."""
    criteria = {
        "B1": {"addendum_wording": 'SaveProof gives zero "verified" in all null runs (s = 0)',
               "hard": True, **b1(table)},
        "B2": {"addendum_wording": 'SaveProof gives zero "verified" at s in {5, 7}%',
               "hard": True, **b2(table)},
        "B3": {"addendum_wording": 'No "verified" in any run where the gate failed',
               "hard": True, **b3(table)},
        "B4": {"addendum_wording": "At least one station passes the gate in the fixed split. If "
               "none does, V5b is declared uninformative, which is not a pass",
               "hard": True, **b4(table)},
        "B5": {"addendum_wording": "Over null runs where SaveProof's gate passed, SaveProof's mean "
               "|savings_pct| is lower than both SEC-ratio's and naive's, in the fixed split and "
               "separately across the rolling windows",
               "hard": False,
               "assumption": "nothing changed at the mine in January; this cannot be verified",
               **b5(table)},
        "B6": {"addendum_wording": 'False "verified" counts for SEC-ratio and naive at s in '
               '{0, 5, 7}; the "verified" rate of all three methods at s in {10, 12, 15}; the '
               "share of runs passing the gate, per station; the stability flag of §10",
               "hard": False, **b6(table)},
    }  # fmt: skip
    return {
        **provenance("v5b", f"{LABEL_NULL} (s = 0); {LABEL_INJECTED} (s > 0)", git_hash, frames),
        "labels": {"null runs (s = 0)": LABEL_NULL, "injected runs (s > 0)": LABEL_INJECTED},
        "addendum": ADDENDUM_PATH,
        "addendum_commit": last_commit(ADDENDUM_PATH),
        "formula": V5B_FORMULA,
        "counts": _counts(table),
        "criteria": criteria,
        "v5b_pass": all(criteria[b]["pass"] for b in ("B1", "B2", "B3", "B4")),
    }


def plot(table: pd.DataFrame) -> Figure:
    """Verified rate (over scored runs, fixed + rolling) vs injected saving, per method, one panel
    per station."""
    scored = table[table["status"] == "ok"]
    stations = list(table["station"].unique())
    fig = Figure(figsize=(14, 4.5), layout="constrained")
    axes = fig.subplots(1, len(stations), sharey=True, squeeze=False)[0]
    for ax, station in zip(axes, stations, strict=True):
        rows = scored[scored["station"] == station]
        for method in METHODS:
            rate = (
                (rows[VERDICT_COLUMNS[method]] == "verified")
                .groupby(rows["injected_saving_pct"])
                .mean()
            )
            ax.plot(rate.index, rate.to_numpy(), marker="o", label=method)
        ax.axvline(100 * ADEETIE_THRESHOLD, color="red", linestyle="--", label="10% (ADEETIE)")
        n_runs = int((table["station"] == station).sum())
        ax.set_title(f"{station}: {V5B_FORMULA}\n{len(rows)} of {n_runs} runs scored")
        ax.set_xlabel("injected saving (%)")
        ax.set_ylabel("share of runs verified")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"V5b, Laayati mine: verified rate vs s (s = 0: {LABEL_NULL}; s > 0: {LABEL_INJECTED})"
    )
    return fig


def main() -> None:
    git_hash = current_git_hash()
    frames = load_stations()
    table = run(frames)
    summary = summarise(table, frames, git_hash)
    for path in write_outputs("v5b", table, summary, plot(table)):
        print(path)
    for name in ("B1", "B2", "B3", "B4", "B5"):
        print(f"{name} pass: {summary['criteria'][name]['pass']}")
    print(f"V5b pass (B1-B4): {summary['v5b_pass']}; addendum commit {summary['addendum_commit']}")


if __name__ == "__main__":
    main()
