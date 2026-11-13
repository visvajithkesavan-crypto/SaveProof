"""Post-hoc secondary analysis, NOT pre-registered: the baseline stability flag of
docs/validation_addendum_v5b.md §10 applied to the V5 windows (validation_plan.md §3-§4).

The method and thresholds were fixed in the addendum before this was computed on V5. It does not
amend, replace or re-score V5 (addendum §1). Every output carries LABEL. No randomness, no
arguments. From the repo root:
    .\\.venv\\Scripts\\python.exe -m experiments.v5_stability_posthoc
"""

import pandas as pd
from matplotlib.figure import Figure

from experiments.common import (
    all_windows,
    current_git_hash,
    last_commit,
    load_datasets,
    provenance,
    window_columns,
    write_outputs,
)
from saveproof.config import STABILITY_DRIFT_SHARE_MIN, STABILITY_P_MAX
from saveproof.validation import SPECS, window_stability

LABEL = "post-hoc secondary analysis, not pre-registered"
ADDENDUM_PATH = "docs/validation_addendum_v5b.md"


def run(windows: dict) -> pd.DataFrame:
    """One row per V5 spec and window: the stability flag of the spec's baseline fit."""
    rows = []
    for spec, (dataset, formula) in SPECS.items():
        for window in windows[dataset]:
            rows.append(
                {**window_columns(spec, window), **window_stability(window.baseline, formula)}
            )
    return pd.DataFrame(rows).assign(label=LABEL)


def summarise(table: pd.DataFrame, frames: dict, git_hash: str) -> dict:
    """Provenance, thresholds and per-spec flag counts (fixed split and rolling windows)."""
    per_spec = {}
    for spec, group in table.groupby("spec", sort=False):
        fixed = group.loc[group["window"] == "fixed", "stability_flagged"]
        rolling = group.loc[group["window"] != "fixed", "stability_flagged"]
        per_spec[spec] = {
            "windows": len(group),
            "not_fitted": int(group["stability_flagged"].isna().sum()),
            "fixed_flagged": None if fixed.empty or pd.isna(fixed.iloc[0]) else bool(fixed.iloc[0]),
            "rolling_flagged": int(rolling.eq(True).sum()),
            "rolling_windows": len(rolling),
        }
    return {
        **provenance("v5_stability_posthoc", LABEL, git_hash, frames),
        "addendum": ADDENDUM_PATH,
        "addendum_commit": last_commit(ADDENDUM_PATH),
        "method": "addendum §10: residuals in time order regressed on days since baseline start, "
        "Newey-West (Bartlett, lag floor(4 (n/100)^(2/9)), no small-sample correction, normal), "
        "two-sided",
        "thresholds": {"p_max": STABILITY_P_MAX, "drift_share_min": STABILITY_DRIFT_SHARE_MIN},
        "role": "reported only; not evidence for or against V5, which stays FAILED (addendum §1)",
        "per_spec": per_spec,
    }


def plot(table: pd.DataFrame) -> Figure:
    """Drift share per window (x = baseline start), one panel per spec; flagged windows filled."""
    fig = Figure(figsize=(13, 8), layout="constrained")
    axes = fig.subplots(2, 2)
    for ax, (spec, (_, formula)) in zip(axes.flat, SPECS.items(), strict=True):
        rows = table[(table["spec"] == spec) & table["stability_flagged"].notna()]
        x = pd.to_datetime(rows["baseline_start"]).to_numpy()
        drift = rows["stability_drift_share"].to_numpy(dtype=float)
        flagged = rows["stability_flagged"].eq(True).to_numpy()
        ax.scatter(
            x[~flagged], drift[~flagged], facecolors="none", edgecolors="C0", label="not flagged"
        )
        ax.scatter(x[flagged], drift[flagged], color="C3", label="flagged (p < 0.05 and D >= 0.05)")
        ax.axhline(STABILITY_DRIFT_SHARE_MIN, color="red", linestyle="--", label="D = 0.05")
        ax.set_title(f"{spec}: {formula}")
        ax.set_xlabel("baseline window start")
        ax.set_ylabel("drift share D")
        ax.legend(fontsize=8)
    fig.suptitle(f"Baseline stability on the V5 windows ({LABEL})")
    return fig


def main() -> None:
    git_hash = current_git_hash()
    frames = load_datasets()
    table = run(all_windows(frames))
    summary = summarise(table, frames, git_hash)
    for path in write_outputs("v5_stability_posthoc", table, summary, plot(table)):
        print(path)
    print(LABEL)


if __name__ == "__main__":
    main()
