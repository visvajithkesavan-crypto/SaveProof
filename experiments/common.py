"""Plumbing shared by the null-test and injected-saving scripts: loading the real datasets with the
R1 loaders, the windows of docs/validation_plan.md §4, git provenance and the results/ files.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from saveproof.ingest.irish import load_irish
from saveproof.ingest.uci import load_uci_steel
from saveproof.validation import (
    ROLLING_DAYS,
    SPECS,
    Window,
    fixed_split,
    rolling_windows,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
PLAN_PATH = "docs/validation_plan.md"


def load_datasets() -> dict[str, pd.DataFrame]:
    """Daily contract frames keyed by the contract `dataset` value, read from data/raw."""
    return {
        "uci_steel": load_uci_steel(
            REPO / "data" / "raw" / "uci_steel" / "Steel_industry_data.csv"
        ),
        "irish_mi_a": load_irish(REPO / "data" / "raw" / "irish"),
    }


def all_windows(frames: dict[str, pd.DataFrame]) -> dict[str, list[Window]]:
    """Fixed split followed by every rolling window, per dataset (§4)."""
    return {
        dataset: [fixed_split(df, dataset), *rolling_windows(df, *ROLLING_DAYS[dataset])]
        for dataset, df in frames.items()
    }


def window_columns(spec: str, window: Window) -> dict:
    """Identifying columns of one results row; dates are plant-local, ends inclusive."""
    dataset, formula = SPECS[spec]
    return {
        "spec": spec,
        "dataset": dataset,
        "formula": formula,
        "window": window.name,
        "baseline_start": window.baseline_start.date().isoformat(),
        "baseline_end": window.baseline_end.date().isoformat(),
        "reporting_start": window.reporting_start.date().isoformat(),
        "reporting_end": window.reporting_end.date().isoformat(),
        "baseline_rows": len(window.baseline),
        "reporting_rows": len(window.reporting),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def last_commit(path: str) -> str:
    """Short hash of the latest commit touching `path` (repo-relative), read at run time."""
    return _git("log", "-1", "--format=%h", "--", path)


def current_git_hash() -> str:
    """Short HEAD hash, with "-dirty" if anything outside results/ is uncommitted or untracked."""
    short = _git("rev-parse", "--short", "HEAD")
    changes = _git("status", "--porcelain", "--", ".", ":(exclude)results")
    return f"{short}-dirty" if changes else short


def provenance(experiment: str, label: str, git_hash: str, frames: dict) -> dict:
    """Header of every summary JSON: what ran, on which code, under which plan, on which data."""
    plan_commits = _git("log", "--format=%h", "--", PLAN_PATH).splitlines()
    return {
        "experiment": experiment,
        "label": label,
        "git_hash": git_hash,
        "plan": PLAN_PATH,
        "plan_commit": plan_commits[0],  # latest commit touching the plan (incl. amendments)
        "plan_signoff_commit": plan_commits[-1],  # commit that added the signed-off plan
        "datasets": {
            dataset: {
                "rows": len(df),
                "first_date": df["period_start"].min().date().isoformat(),
                "last_date": df["period_start"].max().date().isoformat(),
            }
            for dataset, df in frames.items()
        },
    }


def status_counts(table: pd.DataFrame) -> dict:
    """Runs per spec: total, scored ("ok") and excluded ("insufficient_data")."""
    out = {}
    for spec, group in table.groupby("spec", sort=False):
        out[spec] = {
            "runs": len(group),
            "ok": int((group["status"] == "ok").sum()),
            "insufficient_data": int((group["status"] == "insufficient_data").sum()),
        }
    return out


def write_outputs(name: str, table: pd.DataFrame, summary: dict, figure: Figure) -> list[Path]:
    """Write <name>_<hash>.csv, <name>_<hash>_summary.json and <name>_<hash>.png to results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"{name}_{summary['git_hash']}"
    paths = [stem.with_suffix(".csv"), Path(f"{stem}_summary.json"), stem.with_suffix(".png")]
    table.to_csv(paths[0], index=False)
    paths[1].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure.savefig(paths[2], dpi=150, bbox_inches="tight")
    return paths
