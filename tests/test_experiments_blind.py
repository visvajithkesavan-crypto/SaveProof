"""R4 scripts on tiny SIMULATED plant sets written to tmp_path. The real dev/blind sets are never
generated here and no file under data/ is read or written."""

import dataclasses
import inspect
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import experiments.common as common
from experiments import make_plant_set, run_blind_verdicts, score_blind
from saveproof.contract import PlantMeta, validate
from saveproof.validation import TRUTH_COLUMNS, window_stability

SECRET_SEED = 987654321
FORMULAS = {
    "daily": "kwh ~ production + C(day_type)",
    "monthly": "kwh ~ production + operating_days",
}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(make_plant_set, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def formulas(monkeypatch):
    monkeypatch.setattr(run_blind_verdicts, "DAILY_FORMULA", FORMULAS["daily"])
    monkeypatch.setattr(run_blind_verdicts, "MONTHLY_FORMULA", FORMULAS["monthly"])


# ---------- make_plant_set ----------


def test_composition_of_16_fills_every_cell_once():
    cells = make_plant_set.composition(16, np.random.default_rng(0))
    assert Counter(cells) == Counter(
        (s, g, sector)
        for s in ("S1", "S2", "S3", "S4")
        for g in ("daily", "monthly")
        for sector in ("foundry", "dairy_cold_chain")
    )


@pytest.mark.parametrize("n", [50, 250])
def test_composition_is_as_even_as_possible(n):
    cells = make_plant_set.composition(n, np.random.default_rng(0))
    for position in range(3):
        counts = Counter(cell[position] for cell in cells).values()
        assert max(counts) - min(counts) <= n % 16  # strata, granularity, sectors
    strata = Counter(cell[0] for cell in cells).values()
    assert max(strata) - min(strata) <= 1
    for stratum in ("S1", "S4"):  # half daily, half monthly within each stratum
        grains = Counter(g for s, g, _ in cells if s == stratum).values()
        assert max(grains) - min(grains) <= 1


def test_composition_is_shuffled_so_plant_numbers_reveal_nothing():
    cells = make_plant_set.composition(48, np.random.default_rng(0))
    assert [c[0] for c in cells] != [("S1", "S2", "S3", "S4")[i % 4] for i in range(48)]


def test_main_writes_plants_meta_and_truth_without_leaking(data_dir, capsys):
    make_plant_set.main(["--name", "unit", "--seed", str(SECRET_SEED), "--n", "8"])
    processed = data_dir / "processed" / "unit"
    csvs = sorted(processed.glob("plant_*[0-9].csv"))
    metas = sorted(processed.glob("plant_*_meta.json"))
    assert [p.name for p in csvs] == [f"plant_{i:03d}.csv" for i in range(8)]
    assert len(metas) == 8
    assert {p.name for p in processed.iterdir()} == {p.name for p in csvs + metas}

    meta_fields = {f.name for f in dataclasses.fields(PlantMeta)}
    for path in metas:
        assert set(json.loads(path.read_text(encoding="utf-8"))) == meta_fields

    truth = pd.read_csv(data_dir / "unit_truth.csv")
    assert list(truth.columns) == list(TRUTH_COLUMNS)
    assert list(truth["plant_id"]) == [f"plant_{i:03d}" for i in range(8)]

    printed = capsys.readouterr().out
    assert str(SECRET_SEED) not in printed
    for path in data_dir.rglob("*.*"):
        assert str(SECRET_SEED) not in path.read_text(encoding="utf-8")


def test_same_seed_gives_identical_set(tmp_path, monkeypatch):
    for sub in ("a", "b"):
        monkeypatch.setattr(make_plant_set, "DATA_DIR", tmp_path / sub)
        make_plant_set.main(["--name", "unit", "--seed", "5", "--n", "4"])
    for name in ("unit_truth.csv", "processed/unit/plant_002.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_refuses_to_overwrite_an_existing_set(data_dir):
    make_plant_set.main(["--name", "unit", "--seed", "5", "--n", "2"])
    with pytest.raises(FileExistsError):
        make_plant_set.main(["--name", "unit", "--seed", "6", "--n", "2"])


def test_plant_files_round_trip(data_dir):
    make_plant_set.main(["--name", "unit", "--seed", "3", "--n", "4"])
    plants = common.read_plant_set(data_dir / "processed" / "unit")
    assert [meta.plant_id for _, meta in plants] == [f"plant_{i:03d}" for i in range(4)]
    for df, meta in plants:
        pd.testing.assert_frame_equal(df, validate(df))
        assert isinstance(meta.baseline_start, pd.Timestamp)
        assert meta.retrofit_date == meta.reporting_start
        assert set(df["dataset"]) == {meta.plant_id}


# ---------- run_blind_verdicts ----------


def test_run_blind_verdicts_never_references_the_truth_file():
    assert "_truth" not in inspect.getsource(run_blind_verdicts)


def test_formulas_are_the_signed_section_8_values():
    """docs/validation_plan.md §8, 2026-11-14 (§6 formulas), fixed on the dev set."""
    assert run_blind_verdicts.DAILY_FORMULA == "kwh ~ production + C(day_type)"
    assert run_blind_verdicts.MONTHLY_FORMULA == "kwh ~ production + operating_days"


def test_unset_formula_raises(data_dir, monkeypatch):
    monkeypatch.setattr(run_blind_verdicts, "DAILY_FORMULA", None)
    monkeypatch.setattr(run_blind_verdicts, "MONTHLY_FORMULA", None)
    make_plant_set.main(["--name", "unit", "--seed", "3", "--n", "2"])
    with pytest.raises(ValueError, match="validation_plan.md §6"):
        run_blind_verdicts.run(data_dir / "processed" / "unit")


@pytest.fixture
def verdicts(data_dir, formulas):
    make_plant_set.main(["--name", "unit", "--seed", "3", "--n", "8"])
    return run_blind_verdicts.run(data_dir / "processed" / "unit")


def test_verdicts_have_one_row_per_plant_for_all_three_methods(verdicts):
    assert list(verdicts["plant_id"]) == [f"plant_{i:03d}" for i in range(8)]
    expected = {"status", "savings_pct", "verdict", "band_low_pct", "band_high_pct",
                "naive_savings_pct", "naive_verdict", "bee_sec_savings_pct", "bee_sec_verdict",
                "granularity", "formula", "label"}  # fmt: skip
    assert expected <= set(verdicts.columns)
    assert set(verdicts["label"]) == {"simulated"}
    for granularity, formula in FORMULAS.items():
        rows = verdicts[verdicts["granularity"] == granularity]
        assert len(rows) == 4 and set(rows["formula"]) == {formula}
    assert set(verdicts["status"]) == {"ok"}


def test_every_plant_row_reports_baseline_stability(data_dir, formulas):
    """stability_* columns are window_stability of that plant's baseline (plan §8, 2026-11-14)."""
    make_plant_set.main(["--name", "unit", "--seed", "3", "--n", "8"])
    plants = common.read_plant_set(data_dir / "processed" / "unit")
    table = run_blind_verdicts.run(data_dir / "processed" / "unit")
    keys = (
        "stability_slope_kwh_per_day",
        "stability_p_value",
        "stability_drift_share",
        "stability_flagged",
    )
    assert set(keys) <= set(table.columns)
    for (df, meta), (_, row) in zip(plants, table.iterrows(), strict=True):
        baseline = df[df["period_start"].between(meta.baseline_start, meta.baseline_end)]
        expected = window_stability(baseline, row["formula"])
        for key, value in expected.items():
            got = row[key]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                assert pd.isna(got)
            elif isinstance(value, float):
                assert got == pytest.approx(value, rel=1e-12)
            else:
                assert got == value


def test_main_names_the_verdicts_file_with_the_git_hash(data_dir, formulas, tmp_path, monkeypatch):
    make_plant_set.main(["--name", "unit", "--seed", "3", "--n", "2"])
    monkeypatch.setattr(run_blind_verdicts, "PROCESSED_DIR", data_dir / "processed")
    monkeypatch.setattr(run_blind_verdicts, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_blind_verdicts, "current_git_hash", lambda: "abc1234")
    run_blind_verdicts.main(["--name", "unit"])
    assert len(pd.read_csv(tmp_path / "results" / "unit_verdicts_abc1234.csv")) == 2


# ---------- score_blind ----------


def test_score_reproduces_a_hand_checked_plant(verdicts, data_dir):
    truth = pd.read_csv(data_dir / "unit_truth.csv")
    long, metrics, summary = score_blind.score(verdicts, truth)
    plant = verdicts.iloc[0]
    true_pct = truth.set_index("plant_id").at[plant["plant_id"], "true_savings_pct"]
    row = long[(long["plant_id"] == plant["plant_id"]) & (long["method"] == "bee_sec")].iloc[0]
    assert row["savings_pct"] - row["true_savings_pct"] == pytest.approx(
        plant["bee_sec_savings_pct"] - true_pct
    )
    overall = metrics[(metrics["method"] == "naive") & (metrics["group"] == "overall")].iloc[0]
    naive_error = (
        verdicts["naive_savings_pct"]
        - truth.set_index("plant_id").loc[verdicts["plant_id"], "true_savings_pct"].to_numpy()
    )
    assert overall["mae_pp"] == pytest.approx(naive_error.abs().mean())
    assert set(summary["criteria"]) == {"P1", "P2", "P3", "P4", "P5"}
    assert summary["label"] == "simulated"
    json.dumps(summary, allow_nan=False)


def test_score_main_writes_table_and_summary(verdicts, data_dir, tmp_path, monkeypatch):
    verdicts_path = tmp_path / "unit_verdicts_abc1234.csv"
    verdicts.to_csv(verdicts_path, index=False)
    monkeypatch.setattr(score_blind, "RESULTS_DIR", tmp_path / "results")
    score_blind.main(
        ["--verdicts", str(verdicts_path), "--truth", str(data_dir / "unit_truth.csv")]
    )
    out = tmp_path / "results"
    assert (out / "unit_verdicts_abc1234_score.csv").exists()
    summary = json.loads((out / "unit_verdicts_abc1234_score_summary.json").read_text("utf-8"))
    assert summary["verdicts_file"] == "unit_verdicts_abc1234.csv"
