"""Run SaveProof, naive and BEE SEC-ratio on every plant of a simulated set (docs/validation_plan.md
§6, procedure step 2). Label: simulated.

    .\\.venv\\Scripts\\python.exe -m experiments.run_blind_verdicts --name dev

Reads only data/processed/<name>/ (contract tables and PlantMeta); it never opens the truth.
Baseline and reporting rows are split on PlantMeta's dates. Writes
results/<name>_verdicts_<githash>.csv, one row per plant.
"""

import argparse

import pandas as pd

from experiments.common import PROCESSED_DIR, RESULTS_DIR, current_git_hash, read_plant_set
from saveproof.contract import PlantMeta
from saveproof.engine.baseline import add_calendar_terms, is_monthly
from saveproof.validation import compare_methods, run_window, window_stability

# Chosen on the dev set (seed 1, 50 plants) and fixed before the blind set exists.
# docs/validation_plan.md §8, 2026-11-14 (§6 formulas).
DAILY_FORMULA: str | None = "kwh ~ production + C(day_type)"
# docs/validation_plan.md §8, 2026-11-14 (§6 formulas).
MONTHLY_FORMULA: str | None = "kwh ~ production + operating_days"
LABEL = "simulated"


def formula_for(granularity: str) -> str:
    formula = {"daily": DAILY_FORMULA, "monthly": MONTHLY_FORMULA}[granularity]
    if formula is None:
        raise ValueError(
            f"the {granularity} formula is not set: choose it on the dev set and write it into "
            "docs/validation_plan.md §6 before the blind run"
        )
    return formula


def plant_row(df: pd.DataFrame, meta: PlantMeta) -> dict:
    """SaveProof, both comparators, and the baseline stability flag on one plant."""
    granularity = "monthly" if is_monthly(add_calendar_terms(df)) else "daily"
    formula = formula_for(granularity)
    baseline = df[df["period_start"].between(meta.baseline_start, meta.baseline_end)]
    reporting = df[df["period_start"].between(meta.reporting_start, meta.reporting_end)]
    return {
        "plant_id": meta.plant_id,
        "label": LABEL,
        "granularity": granularity,
        "formula": formula,
        **run_window(baseline, reporting, formula),
        **compare_methods(baseline, reporting, formula),
        **window_stability(baseline, formula),
    }


def run(directory) -> pd.DataFrame:
    """One verdicts row per plant in directory."""
    return pd.DataFrame([plant_row(df, meta) for df, meta in read_plant_set(directory)])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    table = run(PROCESSED_DIR / args.name)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{args.name}_verdicts_{current_git_hash()}.csv"
    table.to_csv(path, index=False)
    print(path)
    print(f"{len(table)} plants; SaveProof status: {table['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
