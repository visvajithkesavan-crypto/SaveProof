"""Tests for the V5b part of saveproof/validation.py: windows (addendum §6), per-run rows with
comparators and the stability flag (§7, §10), and criteria B1-B6 (§8) of
docs/validation_addendum_v5b.md.

All frames and results tables are small, synthetic and built here; rng = np.random.default_rng(42).
No Laayati file is read.
"""

import numpy as np
import pandas as pd
import pytest

from saveproof.engine.baseline import fit_baseline
from saveproof.engine.comparators import bee_sec_savings, naive_savings
from saveproof.engine.savings import PROXY_GATE_FAILURE
from saveproof.engine.stability import baseline_stability
from saveproof.validation import (
    V5B_FORMULA,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    compare_methods,
    run_v5b_window,
    v5b_windows,
    window_stability,
)


def day(text):
    return pd.Timestamp(text)


def mine_frame(start="2018-12-31", n=32, rng=None, unit="process_kpi:THE", noise_kwh=200.0):
    """Daily station frame: kwh = 5000 + 2*production + noise (CV(RMSE) ~ 3%)."""
    rng = rng if rng is not None else np.random.default_rng(42)
    dates = pd.date_range(start, periods=n)
    production = rng.uniform(500, 1500, n)
    return pd.DataFrame(
        {
            "period_start": dates,
            "period_end": dates,
            "kwh": 5000 + 2 * production + rng.normal(0, noise_kwh, n),
            "production": production,
            "production_unit": unit,
            "operating_days": 1.0,
            "source": "real_meter",
            "dataset": "laayati_destoning",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


# ---------- windows (addendum §6) ----------


def test_formula_is_kwh_on_production():
    assert V5B_FORMULA == "kwh ~ production"  # addendum §5


def test_v5b_windows_are_the_fixed_split_and_12_rolling_windows():
    windows = v5b_windows(mine_frame())
    assert len(windows) == 13
    fixed, rolling = windows[0], windows[1:]
    assert fixed.name == "fixed"
    assert (fixed.baseline_start, fixed.baseline_end) == (day("2018-12-31"), day("2019-01-19"))
    assert (fixed.reporting_start, fixed.reporting_end) == (day("2019-01-20"), day("2019-01-31"))
    assert (len(fixed.baseline), len(fixed.reporting)) == (20, 12)
    assert [w.name for w in rolling] == [f"rolling_{i:02d}" for i in range(12)]
    for i, window in enumerate(rolling):
        assert window.baseline_start == day("2018-12-31") + pd.Timedelta(days=i)
        assert (len(window.baseline), len(window.reporting)) == (14, 7)
    last = rolling[-1]
    assert (last.baseline_start, last.baseline_end) == (day("2019-01-11"), day("2019-01-24"))
    assert (last.reporting_start, last.reporting_end) == (day("2019-01-25"), day("2019-01-31"))


@pytest.mark.parametrize(
    ("start", "n"), [("2018-12-31", 31), ("2019-01-01", 32), ("2018-12-30", 33)]
)
def test_v5b_windows_reject_a_frame_that_is_not_the_32_frozen_days(start, n):
    with pytest.raises(ValueError, match="2018-12-31"):
        v5b_windows(mine_frame(start=start, n=n))


def test_v5b_windows_reject_a_missing_day():
    frame = mine_frame().drop(index=10)
    with pytest.raises(ValueError, match="32"):
        v5b_windows(frame)


# ---------- per-window stability and per-run rows ----------

STABILITY_KEYS = {
    "stability_slope_kwh_per_day",
    "stability_p_value",
    "stability_drift_share",
    "stability_flagged",
}


def test_window_stability_is_baseline_stability_of_the_fitted_model():
    baseline = mine_frame().iloc[:20]
    expected = baseline_stability(fit_baseline(baseline, V5B_FORMULA))
    assert window_stability(baseline, V5B_FORMULA) == {
        "stability_slope_kwh_per_day": expected.slope_kwh_per_day,
        "stability_p_value": expected.p_value,
        "stability_drift_share": expected.drift_share,
        "stability_flagged": expected.flagged,
    }


def test_window_stability_is_none_when_the_baseline_cannot_be_fitted():
    baseline = mine_frame().iloc[:20].copy()
    baseline.loc[:9, "event"] = "shutdown"  # 10 of 20 excluded > MAX_EXCLUDED_SHARE
    assert window_stability(baseline, V5B_FORMULA) == dict.fromkeys(STABILITY_KEYS)


def test_compare_methods_scores_both_comparators_on_the_same_frames():
    frame = mine_frame()
    baseline, reporting = frame.iloc[:20], frame.iloc[20:]
    naive = naive_savings(baseline, reporting, V5B_FORMULA)
    bee = bee_sec_savings(baseline, reporting)
    assert compare_methods(baseline, reporting, V5B_FORMULA) == {
        "naive_savings_pct": naive.savings_pct,
        "naive_verdict": naive.verdict,
        "bee_sec_savings_pct": bee.savings_pct,
        "bee_sec_verdict": bee.verdict,
    }


def test_run_v5b_window_passes_the_gate_on_measured_production():
    frame = mine_frame()
    row = run_v5b_window(frame.iloc[:20], frame.iloc[20:], V5B_FORMULA)
    assert row["status"] == "ok"
    assert row["gate_failures"] == ""
    assert row["gate_passed"] is True
    assert {"naive_verdict", "bee_sec_verdict", "verdict"} <= set(row)


def test_run_v5b_window_gate_fails_on_a_proxy_unit():
    frame = mine_frame(unit="proxy:kvarh_lagging")
    row = run_v5b_window(frame.iloc[:20], frame.iloc[20:], V5B_FORMULA)
    assert row["gate_passed"] is False
    assert PROXY_GATE_FAILURE in row["gate_failures"]
    assert row["verdict"] == "likely_not_proven"


def test_run_v5b_window_insufficient_data_has_no_gate_value():
    frame = mine_frame()
    baseline = frame.iloc[:20].copy()
    baseline.loc[:9, "event"] = "shutdown"
    row = run_v5b_window(baseline, frame.iloc[20:], V5B_FORMULA)
    assert row["status"] == "insufficient_data"
    assert row["gate_passed"] is None


# ---------- criteria B1-B6 over hand-made results tables (addendum §8) ----------

COLUMNS = [
    "station",
    "window",
    "injected_saving_pct",
    "status",
    "verdict",
    "gate_passed",
    "savings_pct",
    "bee_sec_savings_pct",
    "bee_sec_verdict",
    "naive_savings_pct",
    "naive_verdict",
    "stability_flagged",
]


def run(station="destoning", window="fixed", s=0, verdict="below", gate=True, ours=0.0,
        bee=0.0, naive=0.0, flagged=False, status="ok"):  # fmt: skip
    """One hand-made results row; comparator verdicts follow the 10% rule from their savings."""
    if status == "insufficient_data":
        verdict = gate = ours = flagged = None
    return {
        "station": station,
        "window": window,
        "injected_saving_pct": s,
        "status": status,
        "verdict": verdict,
        "gate_passed": gate,
        "savings_pct": ours,
        "bee_sec_savings_pct": bee,
        "bee_sec_verdict": "verified" if bee >= 10 else "below",
        "naive_savings_pct": naive,
        "naive_verdict": "verified" if naive >= 10 else "below",
        "stability_flagged": flagged,
    }


def table(*rows):
    return pd.DataFrame(list(rows), columns=COLUMNS)


def test_b1_passes_with_no_verified_null_run_and_ignores_injected_runs():
    t = table(
        run(s=0, verdict="below"),
        run(s=0, verdict="likely_not_proven", gate=False),
        run(s=0, status="insufficient_data"),
        run(s=15, verdict="verified"),  # not a null run
    )
    assert b1(t) == {"pass": True, "n_verified": 0, "n_scored": 2, "n_insufficient": 1}


def test_b1_fails_on_one_verified_null_run():
    t = table(run(s=0, verdict="verified"), run(s=0, verdict="below"))
    assert b1(t) == {"pass": False, "n_verified": 1, "n_scored": 2, "n_insufficient": 0}


@pytest.mark.parametrize(("s", "passes"), [(5, False), (7, False), (0, True), (10, True)])
def test_b2_fails_only_on_verified_at_5_or_7(s, passes):
    t = table(run(s=s, verdict="verified"), run(s=7, verdict="likely_not_proven"))
    assert b2(t)["pass"] is passes


def test_b3_fails_only_when_a_gate_failed_run_verifies():
    passing = table(
        run(s=15, verdict="verified", gate=True),  # gate passed: judged by B1/B2
        run(s=15, verdict="likely_not_proven", gate=False),
        run(s=15, status="insufficient_data"),
    )
    assert b3(passing) == {"pass": True, "n_verified": 0, "n_scored": 1, "n_insufficient": 1}
    failing = table(run(s=12, verdict="verified", gate=False))
    assert b3(failing)["pass"] is False


def test_b4_informative_when_one_station_passes_the_gate_in_the_fixed_null_run():
    t = table(
        run("destoning", "fixed", 0, gate=True),
        run("screening", "fixed", 0, gate=False, verdict="likely_not_proven"),
        run("loading", "fixed", 0, status="insufficient_data"),
        run("loading", "rolling_00", 0, gate=True),  # rolling windows do not count
    )
    assert b4(t) == {
        "pass": True,
        "outcome": "informative",
        "stations_passing_gate": ["destoning"],
    }


def test_b4_uninformative_when_no_station_passes_the_gate_in_the_fixed_split():
    t = table(
        run("destoning", "fixed", 0, gate=False, verdict="likely_not_proven"),
        run("screening", "fixed", 0, gate=False, verdict="likely_not_proven"),
        run("loading", "fixed", 0, status="insufficient_data"),
        run("destoning", "rolling_03", 0, gate=True),
        run("screening", "fixed", 10, gate=True),  # not a null run
    )
    assert b4(t) == {"pass": False, "outcome": "uninformative", "stations_passing_gate": []}


def test_b5_passes_when_saveproof_has_the_lowest_mean_abs_savings_in_both_splits():
    """Fixed: ours |2|, |-1| -> 1.5; SEC 4, 3 -> 3.5; naive |-6|, 5 -> 5.5. Rolling: ours 1;
    SEC 2; naive 3. The gate-failed and injected rows are excluded."""
    t = table(
        run("destoning", "fixed", 0, ours=2.0, bee=4.0, naive=-6.0),
        run("screening", "fixed", 0, ours=-1.0, bee=3.0, naive=5.0),
        run("loading", "fixed", 0, gate=False, verdict="likely_not_proven", ours=50.0),
        run("destoning", "fixed", 10, ours=10.0, bee=0.0, naive=0.0),
        run("destoning", "rolling_00", 0, ours=1.0, bee=-2.0, naive=3.0),
    )
    result = b5(t)
    assert result["fixed"] == {
        "pass": True,
        "n_runs": 2,
        "mean_abs_savings_pct": {"saveproof": 1.5, "bee_sec": 3.5, "naive": 5.5},
    }
    assert result["rolling"]["mean_abs_savings_pct"] == {
        "saveproof": 1.0,
        "bee_sec": 2.0,
        "naive": 3.0,
    }
    assert result["pass"] is True


def test_b5_fails_if_either_comparator_is_not_beaten_in_either_split():
    """Fixed: ours 3 vs SEC 2 -> fail even though naive 9 is worse. Rolling passes."""
    t = table(
        run("destoning", "fixed", 0, ours=3.0, bee=2.0, naive=9.0),
        run("destoning", "rolling_00", 0, ours=1.0, bee=5.0, naive=5.0),
    )
    result = b5(t)
    assert result["fixed"]["pass"] is False
    assert result["rolling"]["pass"] is True
    assert result["pass"] is False


def test_b5_ties_fail_because_lower_means_strictly_lower():
    t = table(
        run("destoning", "fixed", 0, ours=2.0, bee=2.0, naive=9.0),
        run("destoning", "rolling_00", 0, ours=1.0, bee=5.0, naive=5.0),
    )
    assert b5(t)["fixed"]["pass"] is False


def test_b5_is_not_evaluable_without_gate_passed_null_runs():
    """No gate-passed fixed run: that split is None; overall None unless the other split fails."""
    t = table(
        run("destoning", "fixed", 0, gate=False, verdict="likely_not_proven"),
        run("destoning", "rolling_00", 0, ours=1.0, bee=5.0, naive=5.0),
    )
    result = b5(t)
    assert result["fixed"] == {"pass": None, "n_runs": 0, "mean_abs_savings_pct": None}
    assert result["pass"] is None
    t_fail = table(
        run("destoning", "fixed", 0, gate=False, verdict="likely_not_proven"),
        run("destoning", "rolling_00", 0, ours=9.0, bee=5.0, naive=5.0),
    )
    assert b5(t_fail)["pass"] is False


def b6_table():
    """destoning: fixed + rolling_00, s in {0, 7, 10, 15}; loading: fixed only, s in {0, 10}."""
    return table(
        run("destoning", "fixed", 0, bee=12.0, naive=3.0, flagged=True),
        run("destoning", "fixed", 7, bee=18.0, naive=11.0, flagged=True),
        run("destoning", "fixed", 10, verdict="verified", bee=22.0, naive=14.0, flagged=True),
        run("destoning", "fixed", 15, verdict="verified", bee=25.0, naive=17.0, flagged=True),
        run("destoning", "rolling_00", 0, bee=1.0, naive=-2.0),
        run("destoning", "rolling_00", 7, bee=8.0, naive=5.0),
        run("destoning", "rolling_00", 10, verdict="likely_not_proven", bee=11.0, naive=8.0),
        run("destoning", "rolling_00", 15, verdict="verified", bee=16.0, naive=13.0),
        run("loading", "fixed", 0, gate=False, verdict="likely_not_proven", bee=2.0),
        run("loading", "fixed", 10, status="insufficient_data", bee=12.0, naive=12.0),
    )


def test_b6_false_verified_counts_for_the_comparators_at_0_5_7():
    """s=0: SEC verifies destoning fixed (12); s=7: SEC fixed (18), naive fixed (11)."""
    result = b6(b6_table())["false_verified"]
    assert result["bee_sec"]["0"] == {"n_verified": 1, "n_scored": 3, "n_insufficient": 0}
    assert result["bee_sec"]["5"] == {"n_verified": 0, "n_scored": 0, "n_insufficient": 0}
    assert result["bee_sec"]["7"]["n_verified"] == 1
    assert result["naive"]["0"]["n_verified"] == 0
    assert result["naive"]["7"]["n_verified"] == 1
    assert set(result) == {"bee_sec", "naive"}


def test_b6_verified_rate_per_station_method_and_s():
    """destoning s=10: ours 1 of 2, SEC 2 of 2, naive 1 of 2 (14 yes, 8 no). loading s=10 is
    insufficient data: rate None, counted."""
    rate = b6(b6_table())["verified_rate"]
    assert rate["destoning"]["saveproof"]["10"] == {
        "verified_rate": 0.5,
        "n_scored": 2,
        "n_insufficient": 0,
    }
    assert rate["destoning"]["bee_sec"]["10"]["verified_rate"] == 1.0
    assert rate["destoning"]["naive"]["10"]["verified_rate"] == 0.5
    assert rate["destoning"]["saveproof"]["15"]["verified_rate"] == 1.0
    assert rate["destoning"]["saveproof"]["12"]["verified_rate"] is None
    assert rate["loading"]["saveproof"]["10"] == {
        "verified_rate": None,
        "n_scored": 0,
        "n_insufficient": 1,
    }
    assert "0" not in rate["destoning"]["saveproof"]


def test_b6_gate_pass_share_per_station_over_scored_runs():
    share = b6(b6_table())["gate_pass_share"]
    assert share["destoning"] == {"share": 1.0, "n_scored": 8, "n_insufficient": 0}
    assert share["loading"] == {"share": 0.0, "n_scored": 1, "n_insufficient": 1}


def test_b6_stability_flag_per_station_from_null_runs():
    stability = b6(b6_table())["stability"]
    assert stability["destoning"] == {
        "fixed_flagged": True,
        "rolling_flagged": 0,
        "rolling_windows": 1,
    }
    assert stability["loading"] == {
        "fixed_flagged": False,
        "rolling_flagged": 0,
        "rolling_windows": 0,
    }
