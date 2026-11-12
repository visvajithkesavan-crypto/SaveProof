"""Injected saving (docs/validation_plan.md §5): real noise, known synthetic saving.

On every null window (§4) and spec (§3), cuts reporting kWh by s in {0, 5, 7, 10, 12, 15}% in
two shapes (uniform, weekday_only) and records the verdict. Writes the results table, the criteria
summary (I2 as amended in §8, plus the original wording) and the operating-characteristic chart
to results/. No randomness, no arguments. From the repo root:
    .\\.venv\\Scripts\\python.exe -m experiments.injected_saving
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
from saveproof.config import ADEETIE_THRESHOLD
from saveproof.validation import (
    INJECTED_SAVING_PCTS,
    SHAPES,
    SPECS,
    i1,
    i2_amended,
    i2_original,
    i3,
    inject_saving,
    run_window,
)

LABEL = "injected (real noise, synthetic saving)"


def run(windows: dict) -> pd.DataFrame:
    """One row per spec, window, injected saving and shape."""
    rows = []
    for spec, (dataset, formula) in SPECS.items():
        for window in windows[dataset]:
            for pct in INJECTED_SAVING_PCTS:
                for shape in SHAPES:
                    reporting = inject_saving(window.reporting, formula, pct / 100, shape)
                    rows.append(
                        {
                            **window_columns(spec, window),
                            "injected_saving_pct": pct,
                            "shape": shape,
                            **run_window(window.baseline, reporting, formula),
                        }
                    )
    return pd.DataFrame(rows).assign(label=LABEL)


def plot(table: pd.DataFrame) -> Figure:
    """Verified rate (over scored windows) vs injected saving, per spec and shape."""
    scored = table[table["status"] == "ok"]
    rate = (
        (scored["verdict"] == "verified")
        .groupby([scored["spec"], scored["shape"], scored["injected_saving_pct"]])
        .mean()
    )
    fig = Figure(figsize=(12, 8), layout="constrained")
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    for ax, (spec, (_, formula)) in zip(axes.flat, SPECS.items(), strict=True):
        for shape in SHAPES:
            if (spec, shape) in rate.index.droplevel(2):
                curve = rate.loc[(spec, shape)]
                ax.plot(curve.index, curve.to_numpy(), marker="o", label=shape)
        ax.axvline(100 * ADEETIE_THRESHOLD, color="red", linestyle="--", label="10% (ADEETIE)")
        n_runs = int((table["spec"] == spec).sum())
        n_scored = int((scored["spec"] == spec).sum())
        ax.set_title(f"{spec}: {formula}\n{n_scored} of {n_runs} runs scored")
        ax.set_xlabel("injected saving (%)")
        ax.set_ylabel("share of windows verified")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Injected saving: verified rate vs s (label: injected, real noise, synthetic saving)"
    )
    return fig


def main() -> None:
    git_hash = current_git_hash()
    frames = load_datasets()
    table = run(all_windows(frames))
    summary = {
        **provenance("injected_saving", LABEL, git_hash, frames),
        "counts": status_counts(table),
        "criteria": {
            "I1": {
                "plan_wording": "zero verified at s in {0, 5, 7}%, for both shapes and every spec",
                "hard": True,
                **i1(table),
            },
            "I2": {
                "plan_wording": "any spec that fails the adequacy gate gives zero verified at "
                "every s (amended 2026-11-10, see §8)",
                "evaluated_as": "runs with ashrae_ok False, every s and shape",
                "hard": True,
                **i2_amended(table),
            },
            "I2_original": {
                "plan_wording": "UCI-A and IR-A give zero verified at every s "
                "(signed off 2026-11-09; superseded by the amendment, reported for transparency)",
                "hard": False,
                **i2_original(table),
            },
            "I3": {
                "plan_wording": "the verified rate at s = 10, 12, 15% for UCI-B and IR-B, which "
                "is the power curve (reported here for every spec)",
                "hard": False,
                "per_spec": i3(table),
            },
        },
    }
    for path in write_outputs("injected_saving", table, summary, plot(table)):
        print(path)
    for name in ("I1", "I2", "I2_original"):
        print(f"{name} pass: {summary['criteria'][name]['pass']}")


if __name__ == "__main__":
    main()
