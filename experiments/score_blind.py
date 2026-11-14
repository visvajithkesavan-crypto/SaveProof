"""Score a verdicts file against its truth file (docs/validation_plan.md §6, procedure step 3). The
only code that reads a truth file. Label: simulated.

    .\\.venv\\Scripts\\python.exe -m experiments.score_blind
        --verdicts results/<name>_verdicts_<githash>.csv --truth data/<name>_truth.csv

Writes results/<verdicts stem>_score.csv (metrics per method: overall, by stratum, granularity and
fixed-share band) and results/<verdicts stem>_score_summary.json (criteria P1-P5).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.common import PLAN_PATH, RESULTS_DIR, current_git_hash, last_commit
from saveproof.validation import blind_metrics_table, blind_scoring_table, p1, p2, p3, p4, p5

LABEL = "simulated"
# Wording from docs/validation_plan.md §6; "hard" as the plan marks it.
CRITERIA = {
    "P1": ("SaveProof's wrong-side rate is lower than both naive and SEC-ratio, overall and "
           "within S2 + S3", False, p1),
    "P2": ('False-verified rate ("verified" with true saving < 10%) <= 5%', True, p2),
    "P3": ("Empirical 90% band coverage is within [0.85, 0.97] over the non-abstained plants",
           False, p3),
    "P4": ("SaveProof's MAE is lower than SEC-ratio's in S2 + S3, and no more than 1 pp worse "
           "than SEC-ratio in S1", False, p4),
    "P5": ("|bias| < 1 pp overall, excluding S4", False, p5),
}  # fmt: skip
# How ambiguous §6 wording is read here; to be confirmed by a §8 entry before the blind run.
READINGS = [
    "S4 gets only P2 (§6), so P1 and P3-P5 use S1-S3; P1 'overall' means S1-S3",
    "P2 is judged in S1-S3 and separately in S4; its rate is over plants with true saving < 10%",
    "Wrong-side, abstention and false-verified rates count insufficient-data runs as abstentions",
]


def score(verdicts: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(per plant and method table, metrics table, summary with criteria P1-P5)."""
    long = blind_scoring_table(verdicts, truth)
    criteria = {
        name: {"plan_wording": wording, "hard": hard, **criterion(long)}
        for name, (wording, hard, criterion) in CRITERIA.items()
    }
    summary = {"label": LABEL, "plan": PLAN_PATH, "readings": READINGS, "criteria": criteria}
    return long, blind_metrics_table(long), summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verdicts", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    args = parser.parse_args(argv)
    _, metrics, summary = score(pd.read_csv(args.verdicts), pd.read_csv(args.truth))
    summary.update(
        verdicts_file=args.verdicts.name,
        truth_file=args.truth.name,
        scoring_git_hash=current_git_hash(),
        plan_commit=last_commit(PLAN_PATH),
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"{args.verdicts.stem}_score"
    metrics.to_csv(stem.with_suffix(".csv"), index=False)
    summary_path = Path(f"{stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(stem.with_suffix(".csv"))
    print(summary_path)
    overall = metrics[metrics["group"] == "overall"].set_index("method")
    print(overall[["n", "mae_pp", "bias_pp", "wrong_side_rate", "abstention_rate",
                   "band_coverage"]].to_string())  # fmt: skip
    for name, criterion in summary["criteria"].items():
        print(f"{name} pass: {criterion['pass']}")


if __name__ == "__main__":
    main()
