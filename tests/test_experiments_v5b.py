"""Smoke tests for experiments/v5b.py and experiments/v5_stability_posthoc.py on SYNTHETIC frames.

Neither script's main() is called: main() reads the real Laayati / UCI / Irish files, and V5b is
run once, by a human, from a clean commit (addendum §12). rng = np.random.default_rng(42).
"""

import json

import numpy as np
import pandas as pd
import pytest

import experiments.common as common
from experiments import v5_stability_posthoc, v5b
from saveproof.validation import V5B_SAVING_PCTS


def station_frame(station, rng, unit="process_kpi:THE"):
    """32 days from 2018-12-31: kwh = 5000 + 2*production + noise."""
    dates = pd.date_range("2018-12-31", periods=32)
    production = rng.uniform(500, 1500, 32)
    return pd.DataFrame(
        {
            "period_start": dates,
            "period_end": dates,
            "kwh": 5000 + 2 * production + rng.normal(0, 200, 32),
            "production": production,
            "production_unit": unit,
            "operating_days": 1.0,
            "source": "real_meter",
            "dataset": f"laayati_{station}",
            "coverage_pct": 100.0,
            "event": pd.Series([None] * 32, dtype=object),
        }
    )


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(42)
    return {
        station: station_frame(station, rng) for station in ("destoning", "screening", "loading")
    }


@pytest.fixture(scope="module")
def table(frames):
    return v5b.run(frames)


def test_run_has_39_windows_times_6_savings(table):
    assert len(table) == 234  # addendum §7
    assert table.groupby("station").size().to_dict() == {
        "destoning": 78,
        "screening": 78,
        "loading": 78,
    }
    assert sorted(table["injected_saving_pct"].unique()) == list(V5B_SAVING_PCTS)
    assert (table.groupby(["station", "window"]).size() == 6).all()


def test_rows_are_labelled_real_for_null_and_injected_otherwise(table):
    null = table["injected_saving_pct"] == 0
    assert set(table.loc[null, "label"]) == {"real"}
    assert set(table.loc[~null, "label"]) == {"injected (real noise, synthetic saving)"}


def test_rows_carry_comparators_gate_and_stability(table):
    expected = {
        "verdict",
        "gate_passed",
        "naive_savings_pct",
        "naive_verdict",
        "bee_sec_savings_pct",
        "bee_sec_verdict",
        "stability_flagged",
        "stability_p_value",
        "stability_drift_share",
        "stability_slope_kwh_per_day",
        "formula",
        "baseline_start",
        "reporting_end",
    }
    assert expected <= set(table.columns)
    assert set(table["formula"]) == {"kwh ~ production"}


def test_stability_is_one_value_per_window_whatever_the_injection(table):
    per_window = table.groupby(["station", "window"])["stability_p_value"].nunique()
    assert (per_window == 1).all()


def test_injection_cuts_reporting_kwh_uniformly_for_every_method(table):
    """Same window, s = 10 vs s = 0: actual kWh x 0.9; comparators see the same cut frame."""
    rows = table[(table["station"] == "destoning") & (table["window"] == "fixed")]
    null = rows[rows["injected_saving_pct"] == 0].iloc[0]
    cut = rows[rows["injected_saving_pct"] == 10].iloc[0]
    assert cut["actual_kwh"] == pytest.approx(0.9 * null["actual_kwh"], rel=1e-12)
    # naive expected kWh does not depend on reporting kWh: 1 - 0.9 (1 - x) = 0.1 + 0.9 x
    expected = 100 * (0.1 + 0.9 * null["naive_savings_pct"] / 100)
    assert cut["naive_savings_pct"] == pytest.approx(expected, rel=1e-9)


def test_summary_has_criteria_provenance_and_is_strict_json(table, frames):
    summary = v5b.summarise(table, frames, "abc1234")
    assert set(summary["criteria"]) == {"B1", "B2", "B3", "B4", "B5", "B6"}
    assert summary["git_hash"] == "abc1234"
    assert summary["addendum"] == "docs/validation_addendum_v5b.md"
    assert summary["addendum_commit"] == common.last_commit("docs/validation_addendum_v5b.md")
    assert summary["labels"] == {
        "null runs (s = 0)": "real",
        "injected runs (s > 0)": "injected (real noise, synthetic saving)",
    }
    assert summary["counts"]["destoning"] == {"runs": 78, "ok": 78, "insufficient_data": 0}
    hard = [summary["criteria"][b]["pass"] for b in ("B1", "B2", "B3", "B4")]
    assert summary["v5b_pass"] is all(hard)
    json.dumps(summary, allow_nan=False)


def test_last_commit_is_a_short_hash():
    commit = common.last_commit("docs/validation_addendum_v5b.md")
    assert 7 <= len(commit) <= 12
    int(commit, 16)


def test_plot_has_one_panel_per_station(table):
    figure = v5b.plot(table)
    assert len(figure.axes) == 3
    assert all(len(ax.get_lines()) >= 3 for ax in figure.axes)  # one curve per method


def test_write_outputs_names_files_with_the_hash(table, frames, tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    summary = v5b.summarise(table, frames, "abc1234")
    paths = common.write_outputs("v5b", table, summary, v5b.plot(table))
    assert sorted(p.name for p in paths) == [
        "v5b_abc1234.csv",
        "v5b_abc1234.png",
        "v5b_abc1234_summary.json",
    ]
    assert len(pd.read_csv(tmp_path / "v5b_abc1234.csv")) == 234


# ---------- post-hoc stability on the V5 windows ----------

POSTHOC_LABEL = "post-hoc secondary analysis, not pre-registered"


def v5_frame(dataset, start, n, rng):
    dates = pd.date_range(start, periods=n)
    production = rng.uniform(100, 300, n)
    weekend = (dates.dayofweek >= 5).astype(float)
    return pd.DataFrame(
        {
            "period_start": dates,
            "period_end": dates,
            "kwh": 180 + 0.4 * production - 30 * weekend + rng.normal(0, 5.0, n),
            "production": production,
            "production_unit": "proxy:test",
            "operating_days": 1.0,
            "source": "real_meter",
            "dataset": dataset,
            "coverage_pct": 100.0,
            "event": pd.Series([None] * n, dtype=object),
        }
    )


@pytest.fixture(scope="module")
def posthoc_table():
    rng = np.random.default_rng(42)
    frames = {
        "uci_steel": v5_frame("uci_steel", "2018-01-01", 365, rng),
        "irish_mi_a": v5_frame("irish_mi_a", "2025-05-23", 184, rng),
    }
    return v5_stability_posthoc.run(common.all_windows(frames))


def test_posthoc_has_one_row_per_spec_and_window_with_the_label(posthoc_table):
    """UCI: fixed + 14 rolling (180/90, step 7); Irish 184 days: fixed + 8 rolling (90/45)."""
    counts = posthoc_table.groupby("spec").size().to_dict()
    assert counts == {"UCI-A": 15, "UCI-B": 15, "IR-A": 9, "IR-B": 9}
    assert set(posthoc_table["label"]) == {POSTHOC_LABEL}
    assert {"stability_flagged", "stability_drift_share", "window"} <= set(posthoc_table.columns)


def test_posthoc_summary_and_plot_carry_the_label(posthoc_table):
    summary = v5_stability_posthoc.summarise(posthoc_table, {}, "abc1234")
    assert summary["label"] == POSTHOC_LABEL
    assert summary["addendum_commit"] == common.last_commit("docs/validation_addendum_v5b.md")
    assert set(summary["per_spec"]) == {"UCI-A", "UCI-B", "IR-A", "IR-B"}
    assert summary["per_spec"]["UCI-B"]["windows"] == 15
    json.dumps(summary, allow_nan=False)
    figure = v5_stability_posthoc.plot(posthoc_table)
    assert POSTHOC_LABEL in figure.get_suptitle()
