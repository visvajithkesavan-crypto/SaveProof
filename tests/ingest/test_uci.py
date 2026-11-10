from pathlib import Path

import pandas as pd
import pytest

from saveproof.contract import validate
from saveproof.ingest.uci import load_uci_steel

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_uci_midnight_rows_are_assigned_after_interval_end_adjustment():
    daily = load_uci_steel(FIXTURES / "uci_steel_midnight.csv")
    validate(daily)

    assert list(daily["period_start"]) == list(pd.to_datetime(["2018-01-01", "2018-01-02"]))
    assert daily.loc[0, "kwh"] == pytest.approx(6.0)
    assert daily.loc[0, "production"] == pytest.approx(60.0)
    assert daily.loc[0, "coverage_pct"] == pytest.approx(3 / 96 * 100)
    assert daily.loc[1, "kwh"] == pytest.approx(4.0)
    assert daily.loc[1, "production"] == pytest.approx(40.0)
    assert daily.loc[1, "coverage_pct"] == pytest.approx(1 / 96 * 100)


def test_uci_contract_metadata_and_production_source_are_exact():
    daily = load_uci_steel(FIXTURES / "uci_steel_midnight.csv")

    assert daily["production_unit"].tolist() == ["proxy:kvarh_lagging"] * 2
    assert daily["source"].tolist() == ["real_meter"] * 2
    assert daily["dataset"].tolist() == ["uci_steel"] * 2
    assert daily["operating_days"].tolist() == [1.0, 1.0]
    assert daily["event"].isna().all()
    assert daily["production"].tolist() == [60.0, 40.0]
